"""doesitbite — catch silent no-ops in ML pipeline transforms.

A transform that doesn't actually change its input (a mis-targeted in-place edit, a
dead code path, a clamp that never fires) silently degrades your results while every
test stays green. ``@bites`` asserts a transform MEASURABLY changes its input; use it
in your tests or wrap a suspicious step.

Born 2026-10-28/29: a KV-cache quantizer that mutated the wrong attribute and a
packing optimizer stuck in a local minimum both passed unnoticed until stress-tested.
"""
from __future__ import annotations
import functools


def _meanabs(x):
    try:
        import torch
        if torch.is_tensor(x):
            return (x.float()).abs().mean().item()
    except Exception:
        pass
    try:
        import numpy as np
        a = np.asarray(x, dtype="float64")
        return float(np.abs(a).mean())
    except Exception:
        return None


def changed(before, after) -> float | None:
    """Mean absolute difference between two array/tensor-like objects, or None."""
    try:
        import torch
        if torch.is_tensor(before) and torch.is_tensor(after):
            return (after.float() - before.float()).abs().mean().item()
    except Exception:
        pass
    try:
        import numpy as np
        return float(np.abs(np.asarray(after, "float64") - np.asarray(before, "float64")).mean())
    except Exception:
        return None


def bites(min_change: float = 1e-9):
    """Decorator: assert ``fn(x, ...)`` returns something that differs from ``x``.

    Raises ``AssertionError`` (a silent no-op) when the output equals the input within
    ``min_change``. The first positional arg is treated as the input.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(x, *a, **k):
            try:
                import torch
                before = x.clone() if hasattr(x, "clone") else x
            except Exception:
                before = x
            out = fn(x, *a, **k)
            d = changed(before, out)
            if d is not None and d <= min_change:
                raise AssertionError(
                    f"{fn.__name__} is a SILENT NO-OP: output differs from input by "
                    f"{d:.3g} (<= {min_change:g}). It did not bite.")
            return out
        return wrapper
    return deco


def assert_bites(fn, x, *args, min_change: float = 1e-9, **kwargs):
    """Imperative form: run ``fn(x, ...)`` once and assert it changed ``x``."""
    return bites(min_change)(fn)(x, *args, **kwargs)
