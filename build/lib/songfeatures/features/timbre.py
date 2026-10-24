"""Timbre and production: spectral shape, frequency balance, MFCCs, timbral change."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..audio import HOP, N_FFT, SR, SR_HI, Track
from ..utils import NAN, fnum, linear_slope

NAME = "timbre"
INFO = {
    "tool": "librosa, scipy",
    "reference": "Peeters et al. (2011) Timbre Toolbox, JASA; McFee et al. (2015) librosa",
}

BANDS = {  # long-term average spectrum bands (Hz), standard mixing-engineer ranges
    "sub_bass": (20, 60), "bass": (60, 250), "low_mid": (250, 500), "mid": (500, 2000),
    "high_mid": (2000, 4000), "presence": (4000, 6000), "brilliance": (6000, 20000),
}
N_MFCC = 13

FEATURES = {
    "spectral_centroid_mean_hz": ("Mean spectral centroid; perceived brightness", "Hz"),
    "spectral_centroid_sd_hz": ("SD of spectral centroid over time", "Hz"),
    "spectral_bandwidth_mean_hz": ("Mean spectral bandwidth (spread)", "Hz"),
    "spectral_rolloff85_mean_hz": ("Mean frequency below which 85% of energy lies", "Hz"),
    "spectral_flatness_mean": ("Mean spectral flatness; noise-like (1) vs tonal (0)", "0-1"),
    "spectral_flux_mean": ("Mean frame-to-frame spectral change (L2, normalised spectra)", "a.u."),
    "spectral_contrast_mean_db": ("Mean peak-valley spectral contrast across 7 sub-bands", "dB"),
    "zero_crossing_rate_mean": ("Mean zero-crossing rate; noisiness/percussiveness", "proportion"),
    "spectral_tilt_db_per_octave": ("Slope of the long-term average spectrum, 50 Hz-16 kHz", "dB/oct"),
    "timbre_change_rate": ("Mean MFCC distance between consecutive beats; timbral variability", "a.u."),
}
for b, (lo, hi) in BANDS.items():
    FEATURES[f"band_share_{b}"] = (f"Share of spectral energy in {lo}-{hi} Hz (long-term average spectrum)", "proportion")
for i in range(1, N_MFCC + 1):
    FEATURES[f"mfcc{i:02d}_mean"] = (f"Mean of MFCC coefficient {i} (timbre envelope)", "a.u.")
    FEATURES[f"mfcc{i:02d}_sd"] = (f"SD of MFCC coefficient {i}", "a.u.")
PATTERNS: list = []


def extract(track: Track, cfg: dict) -> dict:
    import librosa

    out = {k: NAN for k in FEATURES}
    S = track.S
    P = S ** 2
    loud = librosa.power_to_db(P.sum(axis=0), ref=np.max) > -60  # ignore near-silent frames
    if loud.sum() < 10:
        loud[:] = True

    cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0][loud]
    out["spectral_centroid_mean_hz"] = fnum(cen.mean())
    out["spectral_centroid_sd_hz"] = fnum(cen.std())
    out["spectral_bandwidth_mean_hz"] = fnum(librosa.feature.spectral_bandwidth(S=S, sr=SR)[0][loud].mean())
    out["spectral_rolloff85_mean_hz"] = fnum(librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0][loud].mean())
    out["spectral_flatness_mean"] = fnum(librosa.feature.spectral_flatness(S=S)[0][loud].mean())
    out["spectral_contrast_mean_db"] = fnum(librosa.feature.spectral_contrast(S=S, sr=SR)[:, loud].mean())
    out["zero_crossing_rate_mean"] = fnum(librosa.feature.zero_crossing_rate(track.y, frame_length=N_FFT,
                                                                            hop_length=HOP)[0][loud].mean())
    Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-12)
    flux = np.linalg.norm(np.diff(Sn, axis=1), axis=0)
    out["spectral_flux_mean"] = fnum(flux[loud[1:]].mean())

    # Long-term average spectrum at 44.1 kHz for full-band balance
    f, pxx = welch(track.y_hi, fs=SR_HI, nperseg=8192, noverlap=4096)
    band_e = {b: pxx[(f >= lo) & (f < hi)].sum() for b, (lo, hi) in BANDS.items()}
    tot = sum(band_e.values()) + 1e-20
    for b, e in band_e.items():
        out[f"band_share_{b}"] = float(e / tot)
    m = (f >= 50) & (f <= 16000) & (pxx > 0)
    out["spectral_tilt_db_per_octave"] = linear_slope(np.log2(f[m]), 10 * np.log10(pxx[m]))

    mf = track.mfcc[1:N_MFCC + 1]
    for i in range(N_MFCC):
        out[f"mfcc{i + 1:02d}_mean"] = fnum(mf[i, loud].mean())
        out[f"mfcc{i + 1:02d}_sd"] = fnum(mf[i, loud].std())
    mfb = track.beat_sync(mf, aggregate=np.mean)
    if mfb.shape[1] > 2:
        out["timbre_change_rate"] = fnum(np.linalg.norm(np.diff(mfb, axis=1), axis=0).mean())
    return out
