"""Source separation (Demucs) and vocal features: presence, timing, pitch, intonation,
vibrato, melodic motion, arrangement balance (stem energy shares).

If the manifest has a `vocals_path` column (e.g. multitrack datasets), that stem is used
instead of separation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import HOP, N_FFT, SR, SR_HI, Track, decode_audio
from ..utils import NAN, db_amp, fnum, runs

NAME = "vocals"
INFO = {
    "tool": "Demucs v4 htdemucs (source separation); librosa pYIN (f0)",
    "reference": "Rouard, Massa & Defossez (2023) Hybrid Transformers for Music Source Separation, ICASSP; "
                 "Mauch & Dixon (2014) pYIN; Sundberg (1987) vibrato",
}
FEATURES = {
    "separation_source": ("Where the vocal stem came from (demucs model name / provided / cached)", "text"),
    "voice_gate": ("Second detector required for vocal frames (essentia_voice_instrumental or none)", "text"),
    "stem_share_vocals": ("Share of total stem energy in the vocal stem", "proportion"),
    "stem_share_drums": ("Share of total stem energy in the drum stem", "proportion"),
    "stem_share_bass": ("Share of total stem energy in the bass stem", "proportion"),
    "stem_share_other": ("Share of total stem energy in the 'other' (harmony/melody instruments) stem", "proportion"),
    "vocal_presence_ratio": ("Share of the track with active lead/backing vocals", "proportion"),
    "is_instrumental": ("1 if vocal presence < 5%", "0/1"),
    "time_to_first_vocal_s": ("Onset of the first sustained (>= 1 s) vocal passage", "s"),
    "n_vocal_phrases": ("Number of vocal phrases (active runs >= 0.5 s)", "count"),
    "vocal_phrase_mean_s": ("Mean vocal phrase duration", "s"),
    "vocal_sung_share": ("Mean share of vocal-active time inside confident sustained (>=150 ms) pitched notes; high = sung, low = spoken/rapped", "proportion"),
    "speech_like_seconds": ("Seconds of vocal activity classified speech-like (sung_share < threshold in 2-s windows); heuristic, validate", "s"),
    "speech_like_vocal_share": ("Share of vocal-active time that is speech-like (spoken word / rap)", "proportion"),
    "vocal_f0_median_hz":("Median vocal fundamental frequency", "Hz"),
    "vocal_f0_median_midi": ("Median vocal pitch as MIDI number (60 = C4)", "MIDI"),
    "vocal_pitch_range_semitones": ("Vocal pitch range, P95 - P5 of voiced frames", "semitones"),
    "vocal_pitch_sd_semitones": ("SD of vocal pitch", "semitones"),
    "pitch_grid_dev_cents": ("Mean absolute frame deviation from the (tuning-adjusted) equal-tempered grid", "cents"),
    "note_intonation_dev_cents": ("Mean absolute deviation of note centres from the grid; ~0 for pitch-corrected vocals", "cents"),
    "within_note_pitch_sd_cents": ("Mean within-note pitch SD after linear detrending; low = flat/pitch-corrected", "cents"),
    "vibrato_note_share": ("Share of notes >= 0.35 s with periodic 4-8 Hz pitch modulation", "proportion"),
    "vibrato_rate_hz": ("Median vibrato rate of vibrato notes", "Hz"),
    "vibrato_extent_cents": ("Median peak-to-peak vibrato extent", "cents"),
    "n_vocal_notes": ("Number of segmented vocal notes", "count"),
    "note_rate_per_vocal_s": ("Notes per second of vocal activity", "1/s"),
    "melodic_interval_mean_abs": ("Mean absolute interval between successive vocal notes", "semitones"),
    "stepwise_motion_share": ("Share of melodic intervals of 1-2 semitones", "proportion"),
    "repeated_note_share": ("Share of melodic intervals of 0 (repeated pitch)", "proportion"),
    "leap_share": ("Share of melodic intervals >= 5 semitones", "proportion"),
}
PATTERNS: list = []

_MODELS: dict = {}
PYIN_SR, PYIN_HOP = 16000, 256


# ------------------------------------------------------------------------- separation
def _device(cfg):
    dev = cfg.get("device", "auto")
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def demucs_separate(stereo: np.ndarray, cfg: dict) -> dict:
    """Separate a (2, n) 44.1 kHz mix into Demucs stems (dict name -> (2, n))."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    name = cfg.get("demucs_model", "htdemucs")
    dev = _device(cfg)
    if name not in _MODELS:
        repo = cfg.get("demucs_repo")   # local folder with <sig>-<sha>.th + <name>.yaml (offline / mirrored weights)
        m = get_model(name, repo=Path(repo).expanduser()) if repo else get_model(name)
        m.to(dev).eval()
        _MODELS[name] = m
    model = _MODELS[name]
    wav = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))
    if model.samplerate != SR_HI:
        import julius
        wav = julius.resample_frac(wav, SR_HI, model.samplerate)
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        est = apply_model(model, ((wav - mu) / sd)[None], device=dev, shifts=int(cfg.get("demucs_shifts", 1)),
                          split=True, overlap=0.25, progress=False)[0]
    est = est * sd + mu
    if model.samplerate != SR_HI:
        import julius
        est = julius.resample_frac(est, model.samplerate, SR_HI)
    est = est.cpu().numpy()
    n = stereo.shape[1]
    return {src: est[i][:, :n] for i, src in enumerate(model.sources)}


def get_stems(track: Track, cfg: dict) -> dict:
    """Return stems (at least 'vocals'), using provided path > cached files > Demucs."""
    def _load():
        import soundfile as sf
        vp = track.meta.get("vocals_path")
        if isinstance(vp, str) and vp.strip():
            track.put("separation_source", "provided")
            return {"vocals": decode_audio(vp, SR_HI)}
        sdir = Path(track.artifact_dir) / "stems" if track.artifact_dir else None
        if sdir is not None and (sdir / "vocals.flac").exists():
            stems = {}
            for f in sdir.glob("*.flac"):
                a, _ = sf.read(f, always_2d=True, dtype="float32")
                stems[f.stem] = a.T
            track.put("separation_source", "cached")
            return stems
        stems = demucs_separate(track.stereo, cfg)
        track.put("separation_source", cfg.get("demucs_model", "htdemucs"))
        keep = cfg.get("keep_stems", "vocals")
        if sdir is not None and keep != "none":
            sdir.mkdir(parents=True, exist_ok=True)
            for k, v in stems.items():
                if keep == "all" or k == "vocals":
                    sf.write(sdir / f"{k}.flac", v.T, SR_HI, subtype="PCM_16")
        return stems
    return track.get("stems", _load)


def vocal_activity(track: Track, cfg: dict) -> dict:
    """Frame-level vocal activity (22.05 kHz frame grid) and the mono vocal stem."""
    def _compute():
        import librosa
        from scipy.ndimage import median_filter
        stems = get_stems(track, cfg)
        v = stems["vocals"]
        v_mono_hi = v.mean(axis=0) if v.ndim == 2 else v
        n = min(len(v_mono_hi), len(track.y_hi))
        v_mono_hi = v_mono_hi[:n]
        v22 = librosa.resample(v_mono_hi, orig_sr=SR_HI, target_sr=SR)
        rv = db_amp(librosa.feature.rms(y=v22, frame_length=N_FFT, hop_length=HOP)[0])
        rm = db_amp(librosa.feature.rms(y=track.y[: len(v22)], frame_length=N_FFT, hop_length=HOP)[0])
        m = min(len(rv), len(rm))
        rel, floor = float(cfg.get("vocal_rel_db", -15.0)), float(cfg.get("vocal_abs_db", -50.0))
        active = (rv[:m] > rm[:m] + rel) & (rv[:m] > floor)
        t = librosa.frames_to_time(np.arange(m), sr=SR, hop_length=HOP)
        gate = voice_gate(track, cfg) if cfg.get("voice_gate", "auto") != "off" else None
        if gate is not None:   # require the voice classifier to hear a voice there (removes stem bleed)
            gt, gp = gate
            active &= np.interp(t, gt, gp) >= float(cfg.get("voice_gate_threshold", 0.5))
            track.put("voice_gate_used", "essentia_voice_instrumental")
        else:
            track.put("voice_gate_used", "none")
        active = median_filter(active.astype(np.int8), size=21).astype(bool)   # ~0.5 s smoothing
        return {"active": active, "t": t, "v_hi": v_mono_hi, "v22": v22}
    return track.get("vocal_activity", _compute)


def voice_gate(track: Track, cfg: dict):
    """Per-patch (~1 s) singing-voice probability of the mix from the Essentia voice/instrumental
    classifier, or None if Essentia models are unavailable. Embeddings are cached on the track and
    re-used by the essentia_models module."""
    try:
        from . import essentia_models as em
        head = next(h for h in em.HEADS if h["name"] == "voice_prob")
        if not (em.model_dir(cfg) / head["file"]).exists() or not (em.model_dir(cfg) / em.EMBEDDERS["effnet"]["file"]).exists():
            return None
        emb = em.effnet_embeddings(track, cfg)
        algo, classes = em._head(head, cfg)
        P = np.asarray(algo(emb.astype(np.float32)))
        p = P[:, classes.index("voice") if "voice" in classes else 0]
    except ImportError:
        return None
    n = len(p)
    tt = (np.arange(n) + 0.5) * track.duration / n
    return tt, p


# ------------------------------------------------------------------------- pitch helpers
def segment_notes(midi: np.ndarray, voiced: np.ndarray, min_frames: int = 5, jump: float = 0.75):
    """Split voiced runs into notes with hysteresis on pitch jumps. Returns (start, end) frame pairs."""
    notes = []
    for a, b in runs(voiced):
        s = a
        center = np.nanmedian(midi[a:min(a + 3, b)])
        off = 0
        for i in range(a, b):
            if abs(midi[i] - center) > jump:
                off += 1
                if off >= 3:
                    e = i - 2
                    if e - s >= min_frames:
                        notes.append((s, e))
                    s, off = e, 0
                    center = np.nanmedian(midi[s:min(s + 3, b)])
            else:
                off = 0
                if i - s < 10:
                    center = np.nanmedian(midi[s:i + 1])
        if b - s >= min_frames:
            notes.append((s, b))
    return notes


def speech_likeness(active: np.ndarray, notes: list, fs: float, threshold: float = 0.2) -> dict:
    """Spoken vs sung vocals. Singing concentrates vocal time in confident, sustained pitched
    notes (>= 150 ms); speech and rap do not (short syllables, gliding/low-salience pitch).
    For 2-s windows of vocal activity, sung_share = share of active frames inside sustained
    notes; windows with sung_share < threshold are speech-like."""
    sustained = np.zeros(len(active), dtype=bool)
    for a, b in notes:
        if (b - a) / fs >= 0.15:
            sustained[a:b] = True
    W = int(2 * fs)
    speech = np.zeros(len(active), dtype=bool)
    shares = []
    for s in range(0, max(len(active) - W, 0) + 1, W // 2):
        act = active[s:s + W]
        if act.mean() < 0.5:
            continue
        sh = float(sustained[s:s + W][act].mean())
        shares.append(sh)
        if sh < threshold:
            speech[s:s + W] |= act
    act_s = active.sum() / fs
    sp_s = float(speech.sum() / fs)
    return {"vocal_sung_share": float(np.mean(shares)) if shares else NAN,
            "speech_like_seconds": sp_s,
            "speech_like_vocal_share": sp_s / act_s if act_s > 0 else NAN}


def vibrato_of_note(cents: np.ndarray, fs: float):
    """Return (rate Hz, peak-to-peak extent cents, periodicity ratio) of one note's pitch curve."""
    t = np.arange(len(cents)) / fs
    resid = cents - np.polyval(np.polyfit(t, cents, 1), t)
    nfft = 1024
    spec = np.abs(np.fft.rfft(resid * np.hanning(len(resid)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= 3) & (f <= 9)
    if spec[1:].sum() <= 0 or not band.any():
        return NAN, NAN, 0.0
    pk = np.argmax(np.where(band, spec, 0))
    near = (f >= f[pk] - 1.0) & (f <= f[pk] + 1.0)
    ratio = spec[near].sum() / spec[1:].sum()
    return float(f[pk]), float(2 * np.sqrt(2) * resid.std()), float(ratio)


# ------------------------------------------------------------------------- extract
def extract(track: Track, cfg: dict) -> dict:
    import librosa

    out = {k: NAN for k in FEATURES}
    stems = get_stems(track, cfg)
    out["separation_source"] = track.get("separation_source", lambda: "unknown")
    if len(stems) >= 4:
        e = {k: float(np.sum(v.astype(np.float64) ** 2)) for k, v in stems.items()}
        tot = sum(e.values()) + 1e-12
        for k in ("vocals", "drums", "bass", "other"):
            if k in e:
                out[f"stem_share_{k}"] = e[k] / tot
    else:
        ev = float(np.sum(stems["vocals"].astype(np.float64) ** 2))
        out["stem_share_vocals"] = ev / (float(np.sum(track.stereo.astype(np.float64) ** 2)) + 1e-12)

    va = vocal_activity(track, cfg)
    act, t = va["active"], va["t"]
    out["voice_gate"] = track.get("voice_gate_used", lambda: "none")
    out["vocal_presence_ratio"] = float(act.mean())
    out["is_instrumental"] = int(act.mean() < 0.05)
    frame_s = HOP / SR
    phrases = [(a, b) for a, b in runs(act) if (b - a) * frame_s >= 0.5]
    long_runs = [(a, b) for a, b in phrases if (b - a) * frame_s >= 1.0]
    out["n_vocal_phrases"] = len(phrases)
    if phrases:
        out["vocal_phrase_mean_s"] = float(np.mean([(b - a) * frame_s for a, b in phrases]))
    if long_runs:
        out["time_to_first_vocal_s"] = float(t[long_runs[0][0]])
    vocal_seconds = act.sum() * frame_s
    track.put("vocal_seconds", vocal_seconds)
    if out["is_instrumental"] == 1:
        out["speech_like_seconds"] = 0.0
        return out
    if not cfg.get("pitch", True):
        return out

    # ---- f0 with pYIN on the vocal stem (16 kHz), masked to vocal-active regions
    v16 = librosa.resample(va["v_hi"], orig_sr=SR_HI, target_sr=PYIN_SR)
    f0, vflag, vprob = librosa.pyin(v16, fmin=65.0, fmax=1100.0, sr=PYIN_SR, frame_length=1024,
                                    hop_length=PYIN_HOP)
    tf = librosa.times_like(f0, sr=PYIN_SR, hop_length=PYIN_HOP)
    act_f = np.interp(tf, t, act.astype(float), left=0, right=0) > 0.5
    voiced = vflag & (vprob > 0.5) & act_f & np.isfinite(f0)
    midi = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
    fs = PYIN_SR / PYIN_HOP
    notes = segment_notes(midi, voiced, min_frames=int(0.08 * fs)) if voiced.any() else []
    out.update(speech_likeness(act_f, notes, fs, float(cfg.get("speech_sung_threshold", 0.2))))
    if voiced.sum() < 50:
        return out
    mv = midi[voiced]
    out["vocal_f0_median_hz"] = float(np.median(f0[voiced]))
    out["vocal_f0_median_midi"] = float(np.median(mv))
    p5, p95 = np.percentile(mv, [5, 95])
    out["vocal_pitch_range_semitones"] = float(p95 - p5)
    out["vocal_pitch_sd_semitones"] = float(mv.std())
    tuning = float(librosa.pitch_tuning(f0[voiced]))            # fraction of a semitone
    dev = 100 * ((mv - tuning) - np.round(mv - tuning))
    out["pitch_grid_dev_cents"] = float(np.mean(np.abs(dev)))

    if notes:
        centers = np.array([np.median(midi[a:b]) for a, b in notes])
        cd = 100 * ((centers - tuning) - np.round(centers - tuning))
        out["note_intonation_dev_cents"] = float(np.mean(np.abs(cd)))
        sds, vib = [], []
        for (a, b), c in zip(notes, centers):
            cents = 100 * (midi[a:b] - c)
            tt = np.arange(b - a) / fs
            if b - a >= 5:
                sds.append(float(np.std(cents - np.polyval(np.polyfit(tt, cents, 1), tt))))
            if (b - a) / fs >= 0.35:
                rate, ext, ratio = vibrato_of_note(cents, fs)
                vib.append((rate, ext, ratio >= 0.4 and ext >= 20 and 4 <= rate <= 8))
        out["within_note_pitch_sd_cents"] = float(np.mean(sds)) if sds else NAN
        if vib:
            flags = np.array([v[2] for v in vib])
            out["vibrato_note_share"] = float(flags.mean())
            if flags.any():
                out["vibrato_rate_hz"] = float(np.median([v[0] for v in vib if v[2]]))
                out["vibrato_extent_cents"] = float(np.median([v[1] for v in vib if v[2]]))
        note_pitch = np.round(centers - tuning).astype(int)
        iv = np.clip(np.diff(note_pitch), -12, 12)
        out["n_vocal_notes"] = len(notes)
        out["note_rate_per_vocal_s"] = float(len(notes) / max(vocal_seconds, 1e-6))
        if iv.size:
            a = np.abs(iv)
            out["melodic_interval_mean_abs"] = float(a.mean())
            out["stepwise_motion_share"] = float(np.mean((a >= 1) & (a <= 2)))
            out["repeated_note_share"] = float(np.mean(a == 0))
            out["leap_share"] = float(np.mean(a >= 5))
            track.sequences["melody_intervals"] = [int(x) for x in iv]
            track.sequences["melody_notes_midi"] = [int(x) for x in note_pitch]
    track.timeseries["vocal_f0_hz"] = np.where(voiced, f0, np.nan).astype(np.float32)
    track.timeseries["vocal_f0_t"] = tf.astype(np.float32)
    return out
