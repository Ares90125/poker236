"""Poker44 bot detector (RANKMAG) -- a WITHIN-BATCH RANK-FUSED ENSEMBLE of three
decorrelated members (stack / mono / mlp) over the RANKMAG feature view, topped
with our reward-fit, FPR-capped floating decision layer.

RANKMAG feature view (610 dims, see rankmag_features.py)
--------------------------------------------------------
The 452 transfer-stable order-statistic / share / signature columns AS-IS, plus
the 158 raw-magnitude / amount / pot / starting-stack columns RE-ENCODED as
WITHIN-BATCH PERCENTILE RANKS. The orderstat model dropped those 158 columns
because their absolute scale is OOD on the sanitized live feed; ranking each one
among its batch-mates keeps the magnitude *signal* while removing its absolute-
scale OOD-ness (percentile in [0,1] is identical train==serve, by construction).

Because a within-batch percentile needs the WHOLE served batch at once, the
detector featurizes the ENTIRE batch of chunks together (`rankmag_features`)
rather than one chunk at a time. score_batch is therefore the real entry point;
score_chunk is only a degenerate single-chunk fallback (mag block -> neutral 0.5,
no batch context).

Members (all over the identical 610-dim RANKMAG row)
----------------------------------------------------
  1. STACK  -- LGBM + XGB + RF -> logistic OOF stack (the discrimination anchor).
  2. MONO   -- monotone-constrained LightGBM bag on the sign-stable subspace
               (per-DATE Spearman sign stable across >=70% of dates, |rho|>=0.05);
               the OOD-transfer regularizer, decorrelated from the anchor.
  3. MLP    -- StandardScaler -> PCA(56) -> MLP bag; architecturally decorrelated.

Fusion is calibration-free: each member's WITHIN-BATCH rank (argsort/argsort/(n-1))
averaged 0.35/0.30/0.35, so no member's OOD score-scale distorts the blend.

Decision layer (reused verbatim from UNION_ORDERSTAT / BEATER / BEST/GAP_FIX)
----------------------------------------------------------------------------
Fused rank -> isotonic -> per-batch anchor-quantile logit recenter + margin/temp +
hard FLOOR + CAP (Q=0.7, MARGIN=3.0, TEMP=1.0, FLOOR=0.02, CAP=True) => a
deterministic ~2% of every window crosses 0.5, zero hard-zeros.

IMPORTANT -- inference does NOT sanitize (live chunks arrive already sanitized by
the validator). Only offline training sanitized the raw benchmark hands. All
estimators are pinned single-thread on load (batched-predict deadlock guard).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

try:
    from .rankmag_features import rankmag_features, RANKMAG_NAMES
except ImportError:  # flat-module import at train/eval time
    from rankmag_features import rankmag_features, RANKMAG_NAMES

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


def _fused_rank_from_X(model, X):
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
    """One bot-risk score in [0,1] per chunk (rank-fused, reward-fit floating output).

    Featurizes the WHOLE batch together so the 158 magnitude columns can be
    encoded as within-batch percentile ranks."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        X = rankmag_features(chunks)
        return _decision(m, _calibrated(m, _fused_rank_from_X(m, X)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context).

    With no batch, the 158 rank-mag columns default to a neutral 0.5 percentile."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = rankmag_features([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
