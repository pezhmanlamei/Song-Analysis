"""Voice quality of the separated lead vocal: openSMILE eGeMAPS v02 (88 functionals:
pitch, jitter, shimmer, HNR, formants, spectral slope, loudness, rate).

Licence note: openSMILE is free for research/education; commercial use needs a licence
from audEERING.
"""
from __future__ import annotations

import numpy as np

from ..audio import HOP, SR, SR_HI, Track
from ..utils import NAN, slug
from .vocals import vocal_activity

NAME = "voice_quality"
INFO = {
    "tool": "openSMILE 3 (eGeMAPSv02 functionals) on vocal-active segments of the Demucs vocal stem",
    "reference": "Eyben et al. (2016) The Geneva Minimalistic Acoustic Parameter Set (GeMAPS), IEEE TAC; "
                 "Eyben et al. (2010) openSMILE",
}
FEATURES = {"voice_quality_seconds": ("Seconds of vocal-active audio analysed", "s")}
PATTERNS = [(r"^egemaps_", "openSMILE eGeMAPSv02 functional of the lead vocal (see Eyben et al. 2016 for definitions)", "varies")]

_SMILE = None


def extract(track: Track, cfg: dict) -> dict:
    global _SMILE
    import librosa
    import opensmile

    va = vocal_activity(track, cfg)
    act = va["active"]
    seconds = float(act.sum() * HOP / SR)
    out = {"voice_quality_seconds": seconds}
    if seconds < float(cfg.get("min_vocal_seconds", 5.0)):
        return out
    # concatenate vocal-active regions (sample mask on the 44.1 kHz stem)
    mask_hi = np.repeat(act, int(HOP * SR_HI / SR))[: len(va["v_hi"])]
    seg = va["v_hi"][: len(mask_hi)][mask_hi.astype(bool)]
    sig = librosa.resample(seg.astype(np.float32), orig_sr=SR_HI, target_sr=16000)
    if _SMILE is None:
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                 feature_level=opensmile.FeatureLevel.Functionals)
    df = _SMILE.process_signal(sig, 16000)
    for col, val in df.iloc[0].items():
        out[f"egemaps_{slug(col)}"] = float(val) if np.isfinite(val) else NAN
    return out
