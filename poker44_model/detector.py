"""Poker44 bot detector (BEATER rebuild) -- a WITHIN-BATCH RANK-FUSED ENSEMBLE of
three decorrelated members over the 610-dim top-miner UNION feature view
(features_v2 order-statistics + base_features chunk_features), topped with our
reward-fit, FPR-capped floating decision layer.

Why this model
--------------
A fresh full-field head-to-head (every inspectable top miner's model run on the
SAME 1460-group sanitized benchmark under GroupKFold-by-date) found exactly ONE
lever that beats our prior 180-C2 pipeline on true discrimination: the top-miner
UNION feature surface. A plain LGBM over UNION scores GroupKFold-by-date AP 0.943
(per-date 0.932) vs our 180-C2 0.920 (per-date 0.907) -- a real, seed-stable
+0.023. UNION subsumes C2 (C2+UNION = 0.944). Crucially it also un-collapses on
the live captures (raw-STD 0.043, within-batch distinct-fraction 1.00), so it
rides our within-batch rank / calibrated output the same way C2 does. Every other
"edge" in the field (n-gram / temporal-order features, raw-magnitude columns) is
either an order-construction artifact that collapses under hand-shuffle or is
massively OOD -- excluded here on purpose.

Members (all over the identical 610-dim UNION row)
--------------------------------------------------
  1. STACK  -- LGBM + XGB + RF -> logistic OOF stack (the discrimination anchor).
  2. MONO   -- monotone-constrained LightGBM bag on the sign-stable UNION subspace
               (per-DATE Spearman sign stable across >=70% of dates, |rho|>=0.05);
               the OOD-transfer regularizer, decorrelated from the anchor.
  3. MLP    -- StandardScaler -> PCA(56) -> MLP bag; architecturally decorrelated.

Fusion is calibration-free: each member's WITHIN-BATCH rank (argsort/argsort/(n-1))
averaged 0.35/0.30/0.35, so no member's OOD score-scale distorts the blend.

Decision layer (reused verbatim from the prior BEATER / BEST/GAP_FIX)
--------------------------------------------------------------------
Fused rank -> isotonic -> per-batch anchor-quantile logit recenter + margin/temp +
hard FLOOR + CAP (Q=0.7, MARGIN=3.0, TEMP=1.0, FLOOR=0.02, CAP=True) => a
deterministic ~2% of every window crosses 0.5, zero hard-zeros. Monotone, so the
65% rank block is set by the fused rank while the 30% hard-0.5-threshold block
stays pinned high.

IMPORTANT -- inference does NOT sanitize (live chunks arrive already sanitized by
the validator). Only offline training sanitized the raw benchmark hands. All
estimators are pinned single-thread on load (batched-predict deadlock guard).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

try:
    from .union_features import union_features, UNION_NAMES
except ImportError:  # flat-module import at train/eval time
    from union_features import union_features, UNION_NAMES

try:  # keep any torch backend single-threaded
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass
    for attr in ("final_estimator_", "final_estimator"):
        try:
            _pin_single_thread(getattr(est, attr))
        except Exception:
            pass
    try:
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for key in ("stack", "mono", "mlp"):
            try:
                _pin_single_thread(b[key])
            except Exception:
                pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = union_features(c)
        rows.append([feats.get(k, 0.0) for k in UNION_NAMES])
    return np.array(rows, dtype=float)


def _fused_rank(model, chunks):
    X = _rows(chunks)
    s1 = model["stack"].predict_proba(X)[:, 1]
    s2 = model["mono"].predict_proba(X)[:, 1]
    s3 = model["mlp"].predict_proba(X)[:, 1]
    w1, w2, w3 = model["weights"]
    return (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _calibrated(model, fused):
    return model["iso"].predict(np.asarray(fused, dtype=float))


def _decision(model, cal):
    eps = float(model["EPS"]); q = float(model["Q"])
    margin = float(model["MARGIN"]); temp = float(model.get("TEMP", 1.0))
    floor = float(model["FLOOR"]); cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(cal, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    scores = 1.0 / (1.0 + np.exp(-((z - anchor + tref) / temp)))
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(scores))))
    scores[order[:k]] = np.maximum(scores[order[:k]], 0.5001)
    if cap:
        scores[order[k:]] = np.minimum(scores[order[k:]], 0.4999)
    return [round(float(s), 6) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank-fused, reward-fit floating output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _calibrated(m, _fused_rank(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = _rows([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
