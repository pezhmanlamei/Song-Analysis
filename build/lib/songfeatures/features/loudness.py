"""Loudness, dynamics and stereo image.

Integrated loudness follows ITU-R BS.1770-4 (via pyloudnorm). Short-term /
momentary loudness and Loudness Range follow EBU Tech 3341/3342, implemented
here on the same K-weighting filters.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from ..audio import SR_HI, Track
from ..utils import NAN, db_amp, db_power, fnum

NAME = "loudness"
INFO = {
    "tool": "pyloudnorm (ITU-R BS.1770-4) + own EBU Tech 3342 implementation; numpy/scipy",
    "reference": "ITU-R BS.1770-4 (2015); EBU Tech 3341/3342 (2016); Steinmetz & Reiss (2021) pyloudnorm",
}
FEATURES = {
    "duration_s": ("Track duration", "s"),
    "loudness_integrated_lufs":("Integrated programme loudness (gated), ITU-R BS.1770-4", "LUFS"),
    "loudness_range_lu": ("Loudness Range: P95 - P10 of gated short-term loudness (EBU Tech 3342); macro-dynamics", "LU"),
    "loudness_shortterm_sd_lu": ("SD of gated short-term (3 s) loudness; macro-dynamic variation", "LU"),
    "loudness_shortterm_max_lufs": ("Maximum short-term (3 s) loudness", "LUFS"),
    "loudness_momentary_max_lufs": ("Maximum momentary (400 ms) loudness", "LUFS"),
    "true_peak_dbtp": ("Maximum true peak (4x oversampled)", "dBTP"),
    "sample_peak_dbfs": ("Maximum absolute sample value", "dBFS"),
    "rms_level_dbfs": ("Whole-track RMS level (mono mix)", "dBFS"),
    "crest_factor_db": ("Sample peak minus RMS level; low values = heavy compression/limiting", "dB"),
    "plr_db": ("Peak-to-loudness ratio: true peak - integrated loudness (loudness-war indicator)", "dB"),
    "clipping_ratio": ("Share of samples at or above 0.999 full scale", "proportion"),
    "stereo_width": ("Side energy / (mid + side energy); 0 = mono, 0.5 = uncorrelated", "0-1"),
    "stereo_correlation": ("Pearson correlation between left and right channels", "-1..1"),
    "is_mono_file": ("1 if the file has a single channel", "0/1"),
}
PATTERNS: list = []

ST_WIN, MOM_WIN, STEP = 3.0, 0.4, 0.1


def _k_weight(x: np.ndarray, rate: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting pre-filter to each channel of (ch, n)."""
    from pyloudnorm.iirfilter import IIRfilter
    shelf = IIRfilter(4.0, 1 / np.sqrt(2), 1500.0, rate, "high_shelf")
    hp = IIRfilter(0.0, 0.5, 38.0, rate, "high_pass")
    return np.stack([hp.apply_filter(shelf.apply_filter(c.astype(np.float64))) for c in x])


def windowed_loudness(track: Track) -> dict:
    """Momentary (0.4 s) and short-term (3 s) loudness every 0.1 s. Cached on the track."""
    def _compute():
        x = track.raw
        z = _k_weight(x, SR_HI)
        p = (z ** 2).sum(axis=0)                     # channel-summed power (G=1 for L/R)
        csum = np.concatenate([[0.0], np.cumsum(p)])
        step = int(STEP * SR_HI)

        def _curve(win_s):
            w = int(win_s * SR_HI)
            starts = np.arange(0, max(len(p) - w, 0) + 1, step)
            ms = (csum[starts + w] - csum[starts]) / w
            times = (starts + w / 2) / SR_HI
            return times, -0.691 + db_power(ms, floor_db=-200)
        tm, lm = _curve(MOM_WIN)
        ts, ls = _curve(ST_WIN)
        return {"t_mom": tm, "mom": lm, "t_st": ts, "st": ls}
    return track.get("windowed_loudness", _compute)


def _gate(values: np.ndarray, rel_lu: float) -> np.ndarray:
    v = values[values > -70.0]
    if v.size == 0:
        return v
    mean_pow = np.mean(10 ** ((v + 0.691) / 10))
    thr = -0.691 + 10 * np.log10(mean_pow) + rel_lu
    return v[v >= thr]


def _true_peak(x: np.ndarray) -> float:
    peak = 0.0
    block, pad = 1 << 18, 64
    for c in x:
        for s in range(0, len(c), block):
            seg = c[max(0, s - pad): s + block + pad].astype(np.float64)
            up = resample_poly(seg, 4, 1)
            peak = max(peak, float(np.max(np.abs(up))))
    return peak


def extract(track: Track, cfg: dict) -> dict:
    import pyloudnorm as pyln

    x = track.raw
    out = {k: NAN for k in FEATURES}
    out["duration_s"] = float(track.duration)
    out["is_mono_file"] = int(track.is_mono)

    meter = pyln.Meter(SR_HI)
    out["loudness_integrated_lufs"] = fnum(meter.integrated_loudness(x.T.astype(np.float64)))

    wl = windowed_loudness(track)
    st_g = _gate(wl["st"], -20.0)
    if st_g.size >= 2:
        p10, p95 = np.percentile(st_g, [10, 95])
        out["loudness_range_lu"] = float(p95 - p10)
        out["loudness_shortterm_sd_lu"] = float(st_g.std())
    out["loudness_shortterm_max_lufs"] = fnum(np.max(wl["st"]))
    out["loudness_momentary_max_lufs"] = fnum(np.max(wl["mom"]))

    sample_peak = float(np.max(np.abs(x)))
    tp = _true_peak(x) if cfg.get("true_peak", True) else sample_peak
    rms = float(np.sqrt(np.mean(track.y_hi.astype(np.float64) ** 2)))
    out["sample_peak_dbfs"] = fnum(db_amp(sample_peak))
    out["true_peak_dbtp"] = fnum(db_amp(tp))
    out["rms_level_dbfs"] = fnum(db_amp(rms))
    out["crest_factor_db"] = out["sample_peak_dbfs"] - out["rms_level_dbfs"]
    out["plr_db"] = out["true_peak_dbtp"] - out["loudness_integrated_lufs"]
    out["clipping_ratio"] = float(np.mean(np.abs(x) >= 0.999))

    if not track.is_mono:
        L, R = x[0].astype(np.float64), x[1].astype(np.float64)
        mid, side = (L + R) / 2, (L - R) / 2
        em, es = np.sum(mid ** 2), np.sum(side ** 2)
        out["stereo_width"] = float(es / (em + es + 1e-12))
        out["stereo_correlation"] = fnum(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else NAN
    else:
        out["stereo_width"], out["stereo_correlation"] = 0.0, 1.0

    # keep a 1 Hz short-term loudness curve for plotting / analysis
    track.timeseries["loudness_st_t"] = wl["t_st"][::10]
    track.timeseries["loudness_st_lufs"] = wl["st"][::10]
    return out
