from __future__ import annotations

import math


def trapezoid_scale(
    value: float,
    *,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    """Map value to [0, 1] with a trapezoid curve defined by a, b, c, d."""
    value = float(value)
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    if not all(math.isfinite(item) for item in (value, a, b, c, d)):
        raise ValueError(f"expected finite values, got {value}, {a}, {b}, {c}, {d}")
    if not a < b <= c < d:
        raise ValueError(f"expected a < b <= c < d, got {a}, {b}, {c}, {d}")

    if value <= a or value >= d:
        return 0.0
    if b <= value <= c:
        return 1.0
    if value < b:
        return (value - a) / (b - a)
    return (d - value) / (d - c)


def linear_growth(value: float, *, a: float, b: float) -> float:
    """Map value linearly from [a, b] to [0, 1], clamped at both ends."""
    value = float(value)
    a = float(a)
    b = float(b)
    if not all(math.isfinite(item) for item in (value, a, b)):
        raise ValueError(f"expected finite values, got {value}, {a}, {b}")
    if not a < b:
        raise ValueError(f"expected a < b, got {a}, {b}")

    if value <= a:
        return 0.0
    if value >= b:
        return 1.0
    return (value - a) / (b - a)


def linear_decay(value: float, *, a: float, b: float) -> float:
    """Map value linearly from [a, b] to [1, 0], clamped at both ends."""
    return 1.0 - linear_growth(value, a=a, b=b)


def exp_saturation(value: float, *, half_score_at: float) -> float:
    """Map value to [0, 1) with score 0.5 at half_score_at."""
    value = float(value)
    half_score_at = float(half_score_at)
    if not all(math.isfinite(item) for item in (value, half_score_at)):
        raise ValueError(f"expected finite values, got {value}, {half_score_at}")
    if half_score_at <= 0:
        raise ValueError(f"expected half_score_at > 0, got {half_score_at}")

    if value <= 0:
        return 0.0
    return 1.0 - math.exp(-math.log(2.0) * value / half_score_at)
