"""Open-source equivalents of Spotify's 13 audio features (`open_*` columns).

These are NOT Spotify's values (Spotify/Echo Nest algorithms are proprietary); they are
documented open measurements of the same constructs on the same scales (0-1 unless noted),
computed per song with fixed formulas, so a song's value never depends on the rest of the
corpus. Each column falls back to the next-best source when a model is not installed; the
source actually used is recorded per song in `open_sources`.
"""
from __future__ import annotations

import math

import numpy as np

NAME = "open_features"
INFO = {
    "tool": "derived from songfeatures modules (fixed formulas, see songfeatures/derived.py)",
    "reference": "Construct definitions follow the Spotify Web API audio-features documentation; "
                 "validity of Spotify-style features: Vidas et al. (2026) Music Perception; Lee et al. (2021) ISMIR",
}
PITCH_CLASS = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}

# Fixed reference ranges that map each energy component to 0-1 (typical range of commercial music)
ENERGY_COMPONENTS = {
    "loudness": ("loudness_integrated_lufs", -30.0, -5.0, False),    # louder -> more energy
    "onset_rate": ("onset_rate_per_s", 0.0, 8.0, False),             # more events -> more energy
    "brightness": ("_hf_share", 0.0, 0.40, False),                   # energy above 2 kHz
    "density": ("loudness_range_lu", 0.0, 20.0, True),               # compressed/dense -> more energy
    "noisiness": ("spectral_flatness_mean", 0.0, 0.30, False),       # spectral 'entropy'
}

FEATURES = {
    "open_duration_ms": ("Duration (Spotify duration_ms)", "ms"),
    "open_tempo": ("Tempo (Spotify tempo) = tempo_bpm", "BPM"),
    "open_key": ("Key pitch class, 0 = C ... 11 = B, -1 if undetected (Spotify key)", "0-11"),
    "open_mode": ("1 = major, 0 = minor (Spotify mode)", "0/1"),
    "open_loudness": ("Integrated loudness (Spotify loudness is an average dB level; here ITU-R BS.1770 LUFS)", "dB/LUFS"),
    "open_time_signature": ("Beats per bar (Spotify time_signature): Beat This! downbeats if available, else time_signature_est (3 or 4)", "beats"),
    "open_danceability": ("Danceability 0-1: Essentia danceability classifier probability; fallback DFA danceability / 3 (clipped)", "0-1"),
    "open_energy": ("Energy 0-1: mean of 5 signal components (loudness, onset rate, >2 kHz energy share, "
                    "inverse loudness range, spectral flatness; fixed ranges) averaged with model arousal when available", "0-1"),
    "open_energy_signal": ("Signal-only energy composite (the 5 components above)", "0-1"),
    "open_valence": ("Valence 0-1: (DEAM valence - 1) / 8; fallback emoMusic, then CLAP happy-vs-sad", "0-1"),
    "open_acousticness": ("Acousticness 0-1: Essentia acoustic-mood probability; fallback CLAP acoustic-vs-electronic", "0-1"),
    "open_instrumentalness": ("Instrumentalness 0-1: 1 - vocal_presence_ratio / 0.2 (clipped, from separated vocals); fallback 1 - Essentia voice probability", "0-1"),
    "vocal_detectors_disagree": ("1 if the Essentia voice model (p >= .5) and separation (vocal presence >= 5%) disagree on whether the song has vocals; listen to these", "0/1"),
    "open_speechiness": ("Speechiness 0-1: speech-like (spoken/rapped) vocal seconds / duration; fallback CLAP spoken-vs-sung", "0-1"),
    "open_liveness": ("Liveness 0-1: CLAP P(live recording with audience vs studio)", "0-1"),
    "open_sources": ("Source used for each open_* feature for this song", "text"),
}
PATTERNS: list = []


def _f(row: dict, key: str) -> float:
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x))) if math.isfinite(x) else math.nan


def _first(row, options):
    """options: list of (label, function(row) -> value). Return first finite value + label."""
    for label, fn in options:
        v = fn(row)
        if math.isfinite(v):
            return v, label
    return math.nan, "unavailable"


def energy_signal(row: dict) -> float:
    r = dict(row)
    hf = [_f(r, k) for k in ("band_share_high_mid", "band_share_presence", "band_share_brilliance")]
    r["_hf_share"] = sum(hf) if all(math.isfinite(x) for x in hf) else math.nan
    comps = []
    for _, (key, lo, hi, invert) in ENERGY_COMPONENTS.items():
        v = _f(r, key)
        if math.isfinite(v):
            z = _clip01((v - lo) / (hi - lo))
            comps.append(1 - z if invert else z)
    return float(np.mean(comps)) if len(comps) >= 3 else math.nan


def spotify_equivalents(row: dict) -> dict:
    out, src = {}, {}
    dur = _f(row, "duration_s")
    out["open_duration_ms"] = int(round(dur * 1000)) if math.isfinite(dur) else math.nan
    out["open_tempo"] = _f(row, "tempo_bpm")
    tonic = row.get("key_tonic")
    if isinstance(tonic, str):
        out["open_key"] = PITCH_CLASS.get(tonic, -1)
    else:   # -1 = analysed but no key (Spotify convention); NaN = song not analysed
        out["open_key"] = -1 if math.isfinite(_f(row, "key_clarity")) else math.nan
    mode = row.get("key_mode")
    out["open_mode"] = (1 if mode == "major" else 0) if isinstance(mode, str) else math.nan
    out["open_loudness"] = _f(row, "loudness_integrated_lufs")

    out["open_time_signature"], src["time_signature"] = _first(row, [
        ("beat_this_downbeats", lambda r: _f(r, "beats_per_bar")),
        ("periodicity_estimate", lambda r: _f(r, "time_signature_est"))])

    out["open_danceability"], src["danceability"] = _first(row, [
        ("essentia_classifier", lambda r: _f(r, "danceability_model")),
        ("dfa_div3", lambda r: _clip01(_f(r, "danceability_dfa") / 3.0))])

    sig = energy_signal(row)
    out["open_energy_signal"] = sig
    arousal, a_src = _first(row, [
        ("deam_arousal", lambda r: _clip01((_f(r, "deam_arousal_mean") - 1) / 8)),
        ("emomusic_arousal", lambda r: _clip01((_f(r, "emomusic_arousal_mean") - 1) / 8))])
    if math.isfinite(sig) and math.isfinite(arousal):
        out["open_energy"], src["energy"] = (sig + arousal) / 2, f"signal+{a_src}"
    elif math.isfinite(sig):
        out["open_energy"], src["energy"] = sig, "signal"
    else:
        out["open_energy"], src["energy"] = arousal, a_src

    out["open_valence"], src["valence"] = _first(row, [
        ("deam_valence", lambda r: _clip01((_f(r, "deam_valence_mean") - 1) / 8)),
        ("emomusic_valence", lambda r: _clip01((_f(r, "emomusic_valence_mean") - 1) / 8)),
        ("clap_happy_vs_sad", lambda r: _f(r, "clap_ref_happy"))])

    out["open_acousticness"], src["acousticness"] = _first(row, [
        ("essentia_mood_acoustic", lambda r: _f(r, "mood_acoustic")),
        ("clap_acoustic_vs_electronic", lambda r: _f(r, "clap_ref_acoustic"))])

    # Separated vocals are direct evidence; the voice/instrumental classifier can mistake voice-like
    # lead instruments (fiddle, steel guitar) for singing, so it is only the fallback.
    out["open_instrumentalness"], src["instrumentalness"] = _first(row, [
        ("separation_vocal_presence", lambda r: _clip01(1 - _f(r, "vocal_presence_ratio") / 0.2)),
        ("essentia_voice_model", lambda r: _clip01(1 - _f(r, "voice_prob")))])
    vp, sep = _f(row, "voice_prob"), _f(row, "vocal_presence_ratio")
    out["vocal_detectors_disagree"] = (int((vp >= 0.5) != (sep >= 0.05))
                                       if math.isfinite(vp) and math.isfinite(sep) else math.nan)

    out["open_speechiness"], src["speechiness"] = _first(row, [
        ("speech_like_vocal_time", lambda r: _clip01(_f(r, "speech_like_seconds") / _f(r, "duration_s"))),
        ("clap_spoken_vs_sung", lambda r: _f(r, "clap_ref_speech"))])

    out["open_liveness"], src["liveness"] = _first(row, [
        ("clap_live_vs_studio", lambda r: _f(r, "clap_ref_live"))])

    out["open_sources"] = ";".join(f"{k}={v}" for k, v in src.items())
    return out


def add_spotify_equivalents(table):
    import pandas as pd
    derived = pd.DataFrame([spotify_equivalents(r) for r in table.to_dict(orient="records")], index=table.index)
    return pd.concat([table.drop(columns=[c for c in derived.columns if c in table], errors="ignore"), derived], axis=1)
