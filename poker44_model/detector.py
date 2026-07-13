"""Poker44 bot detector -- MEGABAG_OS: an 8-way BAG of decorrelated orderstat
3-member rank-fused ensembles (STACK+MONO+MLP over the 452-dim transfer-stable
UNION order-statistic view). Each sub-ensemble emits a within-batch fused rank;
the 8 sub-ranks are AVERAGED (variance reduction ~1/sqrt(8) on per-round swing),
then topped with the reused reward-fit FPR-capped floating decision layer
(Q=0.7, MARGIN=3.0, TEMP=1.0, FLOOR=0.02, CAP=True): deterministic ~2% cross 0.5,
zero hard-zeros, max ~0.99. Inference does NOT sanitize (live chunks arrive
sanitized). All estimators pinned single-thread (batched-predict deadlock guard).
"""
from __future__ import annotations
import os
import numpy as np
import joblib

try:
    from .union_features import union_features, UNION_NAMES
except ImportError:
    from union_features import union_features, UNION_NAMES

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try: est.set_params(**{attr: 1})
        except Exception: pass
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception: pass
    for attr in ("final_estimator_", "final_estimator"):
        try: _pin_single_thread(getattr(est, attr))
        except Exception: pass
    try:
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception: pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for sub in b["subs"]:
            for key in ("stack", "mono", "mlp"):
                try: _pin_single_thread(sub[key])
                except Exception: pass
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


def _bag_fused(model, chunks):
    X = _rows(chunks)
    acc = None
    for sub in model["subs"]:
        w1, w2, w3 = sub["weights"]
        s1 = sub["stack"].predict_proba(X)[:, 1]
        s2 = sub["mono"].predict_proba(X)[:, 1]
        s3 = sub["mlp"].predict_proba(X)[:, 1]
        f = (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)
        acc = f if acc is None else acc + f
    return acc / len(model["subs"])


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


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
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, m["iso"].predict(_bag_fused(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = _rows([chunk])
        acc = 0.0
        for sub in m["subs"]:
            w = sub["weights"]
            acc += (w[0] * sub["stack"].predict_proba(X)[:, 1]
                    + w[1] * sub["mono"].predict_proba(X)[:, 1]
                    + w[2] * sub["mlp"].predict_proba(X)[:, 1]) / sum(w)
        return round(float(acc[0] / len(m["subs"])), 6)
    except Exception:
        return 0.5
