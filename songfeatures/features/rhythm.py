"""Rhythm and groove: tempo, tempo stability, pulse clarity, onset density,
syncopation (off-beat onset share), percussiveness, danceability (DFA)."""
from __future__ import annotations

import numpy as np

from ..audio import HOP, SR, Track
from ..utils import NAN, entropy_bits, fnum, linear_slope

NAME = "rhythm"
INFO = {
    "tool": "librosa (beat tracking, tempogram, onsets, HPSS); Essentia Danceability (optional)",
    "reference": "Ellis (2007) beat tracking; Lartillot et al. (2008) pulse clarity; "
                 "Fitzgerald (2010) HPSS; Streich & Herrera (2005) DFA danceability; "
                 "Witek et al. (2014) syncopation & groove",
}
FEATURES = {
    "tempo_bpm": ("Tempo from median inter-beat interval", "BPM"),
    "tempo_local_cv": ("Coefficient of variation of local (tempogram) tempo; tempo instability", "ratio"),
    "tempo_drift_bpm_per_min": ("Linear drift of local tempo over the track", "BPM/min"),
    "beat_interval_cv": ("Coefficient of variation of inter-beat intervals", "ratio"),
    "pulse_clarity": ("Peak of normalised onset autocorrelation in the 60-200 BPM lag range", "0-1"),
    "beat_strength": ("Mean onset strength at beats / mean onset strength overall", "ratio"),
    "onset_rate_per_s": ("Detected note/drum onsets per second; event density", "1/s"),
    "ioi_entropy_bits": ("Entropy of inter-onset-interval histogram (20 ms bins); rhythmic irregularity", "bits"),
    "beat_tracker": ("Beat tracking backend used (librosa or beat_this)", "text"),
    "beats_per_bar": ("Beats per bar from Beat This! downbeats (2-7): the bar length whose beat grid best lines up "
                      "with the detected downbeats; needs beat_tracker=beat_this or meter_backend=beat_this", "beats"),
    "beats_per_bar_confidence": ("How well the detected downbeats line up with that bar grid, above chance "
                                 "(1 = all on the grid; low = unclear meter or many spurious downbeats)", "0-1"),
    "meter_backend_error": ("Error message if the optional Beat This! meter backend failed (empty if fine)", "text"),
    "time_signature_est": ("Estimated beats per bar (3 or 4) from beat-level periodicity of bass onsets, onsets and "
                           "chord changes; 4 unless triple meter is clearly stronger (2/4 and 4/4 both give 4)", "3/4"),
    "meter_triple_margin": ("Triple- minus duple-periodicity score behind time_signature_est (>0 favours 3/4)", "r"),
    "offbeat_onset_ratio": ("Strength-weighted share of onsets on 8th/16th off-beat positions; syncopation proxy. "
                            "Depends on correct beat phase (use beat_this for off-beat-heavy styles)", "proportion"),
    "sixteenth_onset_ratio": ("Strength-weighted share of onsets on 16th-note positions", "proportion"),
    "percussive_energy_ratio": ("Percussive / (harmonic + percussive) energy after HPSS", "proportion"),
    "danceability_dfa": ("Essentia detrended-fluctuation danceability (higher = more danceable); NaN if Essentia missing", "0-~3"),
}
PATTERNS: list = []


def _syncopation(onset_t, onset_w, beat_t):
    """Classify each onset by metrical position inside its beat (on-beat, 8th, 16th)."""
    if len(beat_t) < 4 or len(onset_t) == 0:
        return NAN, NAN
    idx = np.searchsorted(beat_t, onset_t, side="right") - 1
    ok = (idx >= 0) & (idx < len(beat_t) - 1)
    if ok.sum() == 0:
        return NAN, NAN
    t, w, i = onset_t[ok], onset_w[ok], idx[ok]
    phase = (t - beat_t[i]) / (beat_t[i + 1] - beat_t[i])
    grid = np.rint(phase * 4) % 4          # 0 = beat, 2 = 8th off-beat, 1/3 = 16ths
    total = w.sum() + 1e-12
    off8 = w[grid == 2].sum()
    off16 = w[(grid == 1) | (grid == 3)].sum()
    return float((off8 + off16) / total), float(off16 / total)


def _lag_corr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 8 or x.std() == 0:
        return NAN
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def estimate_meter(track: Track, margin: float = 0.1):
    """Beats per bar (3 vs 4) from the periodicity of beat-level accents (Gouyon & Herrera 2003
    style): bass-band onsets, all onsets and chord-change novelty, autocorrelated at beat lags
    3/6 (triple) vs 4/8 (duple). Returns (meter, triple-minus-duple margin)."""
    import librosa
    fr = track.beats["frames"]
    if len(fr) < 24:
        return NAN, NAN
    M = librosa.feature.melspectrogram(y=track.y, sr=SR, hop_length=HOP, n_mels=64, fmax=8000)
    f = librosa.mel_frequencies(n_mels=64, fmax=8000)
    low = librosa.onset.onset_strength(S=librosa.power_to_db(M[f <= 200]), sr=SR, hop_length=HOP)
    feats = [librosa.util.sync(low[None], fr, aggregate=np.max)[0],
             librosa.util.sync(track.onset_env[None], fr, aggregate=np.max)[0]]
    C = track.beat_sync(track.chroma)
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-12)
    feats.append(np.r_[0.0, 1 - np.sum(C[:, 1:] * C[:, :-1], axis=0)])
    s3 = np.nanmean([np.nanmean([_lag_corr(x, 3), _lag_corr(x, 6)]) for x in feats])
    s4 = np.nanmean([np.nanmean([_lag_corr(x, 4), _lag_corr(x, 8)]) for x in feats])
    if not (np.isfinite(s3) and np.isfinite(s4)):
        return NAN, NAN
    return (3 if s3 - s4 > margin else 4), float(s3 - s4)


def meter_from_downbeats(beats: np.ndarray, downbeats: np.ndarray, tol: float = 0.07, window: int = 16,
                         candidates=(2, 3, 4, 5, 6, 7)) -> tuple[float, float]:
    """Bar length from beats + downbeats that tolerates spurious downbeats and dropped beats. (Beat This!
    without its DBN post-processing often marks extra downbeats, so the median number of beats between
    downbeats collapses to 1.) Each downbeat is snapped to its beat index; for every pair of downbeats at most
    `window` beats apart, the distance in beats should be a multiple of the bar length m. The share of such
    pairs is rescaled against chance (1/m): lift = (share - 1/m) / (1 - 1/m), so spurious downbeats do not
    favour short bars, and only nearby pairs are used, so a dropped beat does not shift the grid for the rest
    of the song. Clean 4/4 gives lift 1 for both 2 and 4, so the longest bar length within 95% of the best
    lift is chosen. Returns (beats per bar, lift)."""
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    if beats.size < 8 or downbeats.size < 3:
        return NAN, NAN
    # distances in beats from time / local beat period (robust to single extra or missing beats);
    # the local period is the median inter-beat interval of the 9 beats around each downbeat
    ibi = np.diff(beats)
    k = np.clip(np.searchsorted(beats, downbeats), 0, ibi.size - 1)
    period = np.array([np.median(ibi[max(0, i - 4):i + 5]) for i in k])
    x = (downbeats[None, :] - downbeats[:, None]) / period[:, None]
    near = (x > 0.5) & (x <= window + 0.5)
    r = np.round(x[near])
    d = r[np.abs(x[near] - r) <= tol / np.broadcast_to(period[:, None], x.shape)[near]].astype(int)
    if d.size < 3:
        return NAN, NAN
    lift = {m: (np.mean(d % m == 0) - 1 / m) / (1 - 1 / m) for m in candidates}
    best = max(lift.values())
    if best <= 0:
        return NAN, 0.0
    m = max(k for k, v in lift.items() if v >= 0.95 * best)
    return float(m), float(lift[m])


def extract(track: Track, cfg: dict) -> dict:
    import librosa

    out = {k: NAN for k in FEATURES}
    out["time_signature_est"], out["meter_triple_margin"] = estimate_meter(track, float(cfg.get("meter_margin", 0.1)))
    oenv = track.onset_env
    beats = track.beats
    bt = beats["times"]
    out["beat_tracker"] = track.get("beat_backend", lambda: "librosa")
    db, bt_meter = beats.get("downbeats"), bt
    if db is None and cfg.get("meter_backend") == "beat_this":
        # neural downbeats only for the meter; all other rhythm features keep the main beat tracker
        try:
            from ..audio import _beat_this
            bt_meter, db = _beat_this(track.y_hi, 44100, cfg.get("device", "auto"))
        except Exception as e:  # optional backend: never fail the module
            out["meter_backend_error"] = f"{type(e).__name__}: {e}"[:200]
    if db is not None and len(db) >= 3:
        bpb, conf = meter_from_downbeats(bt_meter, db)
        out["beats_per_bar"], out["beats_per_bar_confidence"] = bpb, conf

    ibi = np.diff(bt)
    if ibi.size >= 4:
        # robust period: median of IBIs averaged over 8-beat spans (cancels frame quantisation)
        span = min(8, len(bt) - 1)
        period = np.median((bt[span:] - bt[:-span]) / span)
        out["tempo_bpm"] = float(60.0 / period)
        out["beat_interval_cv"] = float(ibi.std() / ibi.mean())

    local = librosa.feature.tempo(onset_envelope=oenv, sr=SR, hop_length=HOP, aggregate=None)
    lt = librosa.frames_to_time(np.arange(len(local)), sr=SR, hop_length=HOP)
    core = (lt > 5) & (lt < track.duration - 5)
    if core.sum() > 10:
        out["tempo_local_cv"] = float(local[core].std() / (local[core].mean() + 1e-9))
        out["tempo_drift_bpm_per_min"] = linear_slope(lt[core] / 60, local[core])

    o = oenv - oenv.mean()
    ac = librosa.autocorrelate(o, max_size=int(2.0 * SR / HOP))
    if ac[0] > 0:
        ac = ac / ac[0]
        lags = np.arange(len(ac)) * HOP / SR
        rng = (lags >= 0.3) & (lags <= 1.0)
        out["pulse_clarity"] = float(np.clip(ac[rng].max(), 0, 1))
    out["beat_strength"] = float(oenv[beats["frames"][beats["frames"] < len(oenv)]].mean() / (oenv.mean() + 1e-9))

    on_f = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, hop_length=HOP, units="frames")
    on_t = librosa.frames_to_time(on_f, sr=SR, hop_length=HOP)
    out["onset_rate_per_s"] = float(len(on_t) / track.duration)
    ioi = np.diff(on_t)
    ioi = ioi[(ioi > 0) & (ioi < 2.0)]
    if ioi.size > 5:
        out["ioi_entropy_bits"] = entropy_bits(np.histogram(ioi, bins=np.arange(0, 2.02, 0.02))[0])
    out["offbeat_onset_ratio"], out["sixteenth_onset_ratio"] = _syncopation(on_t, oenv[on_f], bt)

    H, P = track.hpss
    eh, ep = np.sum(np.abs(H) ** 2), np.sum(np.abs(P) ** 2)
    out["percussive_energy_ratio"] = float(ep / (eh + ep + 1e-12))

    try:
        import essentia.standard as es
        dance, _ = es.Danceability(sampleRate=44100)(track.y_hi.astype(np.float32))
        out["danceability_dfa"] = fnum(dance)
    except ImportError:
        pass

    track.sequences["beat_times"] = [round(float(b), 3) for b in bt]
    return out
