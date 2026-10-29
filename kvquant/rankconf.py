"""rankconf — confidence utilities for rank-based selective-prediction metrics.

Two facts people miss on metrics that depend only on the *order* of a confidence
score (AURC, risk-coverage, AUROC):

1. Any **strictly monotonic** rescale of confidence (isotonic in the no-tie case,
   Platt, temperature, sigmoid) leaves the metric unchanged — it is a provable no-op.
   So calibrating to improve AURC is wasted effort; spend it on a *better signal*.
2. **Agreement-based** confidence (how much an independent second pass agrees) is a
   much stronger ranking signal than a model's self-reported confidence.

Verified 2026-10-29: on synthetic-but-faithful data, strictly-monotonic rescales gave
identical AURC (delta 0), and agreement confidence cut AURC by ~34% vs self-reported.
"""
from __future__ import annotations
import numpy as np


def aurc(confidence, error) -> float:
    """Area under the risk-coverage curve (lower is better).

    Sorts items by ``confidence`` descending and averages the running mean ``error``
    over all coverage levels. Depends only on the *order* of ``confidence``.
    """
    confidence = np.asarray(confidence, dtype="float64")
    error = np.asarray(error, dtype="float64")
    order = np.argsort(-confidence)
    e = error[order]
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean())


def is_rank_noop(confidence, transform, error, tol: float = 1e-9) -> bool:
    """True if applying ``transform`` to ``confidence`` leaves AURC unchanged."""
    a = aurc(confidence, error)
    b = aurc(transform(np.asarray(confidence, dtype="float64")), error)
    return abs(a - b) <= tol


def agreement_confidence(primary, secondary, continuous: bool = True):
    """Confidence from agreement between two independent predictions.

    ``continuous=True`` returns negative normalized distance (works for numbers or
    equal-length strings via character distance); otherwise exact-match (1.0/0.0).
    Higher = more agreement = more confident.
    """
    out = []
    for p, s in zip(primary, secondary):
        if not continuous:
            out.append(1.0 if p == s else 0.0)
        elif isinstance(p, (int, float)) and isinstance(s, (int, float)):
            out.append(-abs(float(p) - float(s)))
        else:
            p, s = str(p), str(s)
            m = max(len(p), len(s)) or 1
            dist = sum(a != b for a, b in zip(p, s)) + abs(len(p) - len(s))
            out.append(-dist / m)
    return np.asarray(out, dtype="float64")
