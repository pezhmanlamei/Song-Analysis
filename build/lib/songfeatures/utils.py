"""Small numeric helpers shared by all feature modules."""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

NAN = float("nan")
EPS = 1e-12


def db_power(x, floor_db: float = -120.0):
    """10*log10 of a power value (array-safe), clipped at floor_db."""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(10.0 * np.log10(np.maximum(x, EPS)), floor_db)


def db_amp(x, floor_db: float = -120.0):
    """20*log10 of an amplitude value (array-safe), clipped at floor_db."""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(x), EPS)), floor_db)


def fnum(v):
    """Convert numpy scalars / arrays of size 1 to plain python floats (NaN-safe)."""
    if v is None:
        return NAN
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    try:
        v = float(np.asarray(v).reshape(-1)[0]) if np.ndim(v) else float(v)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def entropy_bits(counts: Iterable[float]) -> float:
    """Shannon entropy (bits) of a histogram / count vector."""
    c = np.asarray(list(counts), dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return NAN
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def linear_slope(t, y) -> float:
    """OLS slope of y on t (NaN-safe)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(t[m]) == 0:
        return NAN
    return float(np.polyfit(t[m], y[m], 1)[0])


def series_summary(prefix: str, times, values, duration: float) -> dict:
    """Standard summary of a time series: mean, sd, P10, P90, range, slope/min, peak position (0-1)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    m = np.isfinite(v)
    out = {f"{prefix}_mean": NAN, f"{prefix}_sd": NAN, f"{prefix}_p10": NAN, f"{prefix}_p90": NAN,
           f"{prefix}_range_p90_p10": NAN, f"{prefix}_slope_per_min": NAN, f"{prefix}_peak_position": NAN}
    if m.sum() < 2:
        return out
    t, v = t[m], v[m]
    p10, p90 = np.percentile(v, [10, 90])
    out.update({
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_sd": float(v.std()),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p90": float(p90),
        f"{prefix}_range_p90_p10": float(p90 - p10),
        f"{prefix}_slope_per_min": linear_slope(t / 60.0, v),
        f"{prefix}_peak_position": float(t[int(np.argmax(v))] / max(duration, EPS)),
    })
    return out


def runs(mask) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True runs in a boolean array."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def slug(text: str) -> str:
    """Lower-case identifier safe for column names."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return s or "x"


def safe_id(text: str) -> str:
    """File-system safe track id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "track"
