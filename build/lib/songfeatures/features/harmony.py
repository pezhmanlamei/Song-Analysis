"""Harmony and tonality: key, mode strength, key changes, chords, harmonic
rhythm/complexity, tonal change, sensory roughness, tuning."""
from __future__ import annotations

import numpy as np

from ..audio import Track
from ..utils import NAN, entropy_bits, fnum

NAME = "harmony"
INFO = {
    "tool": "librosa (CQT chroma, tonnetz, Viterbi); own key/chord/roughness implementations",
    "reference": "Krumhansl & Kessler (1982) key profiles; Temperley (1999); "
                 "Harte et al. (2006) tonal centroid; Sethares (1993) / Plomp & Levelt (1965) roughness; "
                 "Mauch & Dixon (2010) chord estimation",
}
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PROFILES = {
    "krumhansl": (np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
                  np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])),
    "temperley": (np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
                  np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])),
}
FEATURES = {
    "chroma_method": ("Chroma representation used (essentia_hpcp or librosa_cqt)", "text"),
    "tuning_method": ("Tuning estimator used (essentia TuningFrequencyExtractor or librosa)", "text"),
    "key_method": ("Global key estimator used", "text"),
    "key_tonic": ("Estimated tonic pitch class (global)", "C..B"),
    "key_mode": ("Estimated mode (global)", "major/minor"),
    "key_label": ("Estimated key, e.g. 'A minor'", "text"),
    "key_clarity": ("Correlation of chroma with the best key profile; tonal clarity", "-1..1"),
    "key_margin": ("Best minus second-best key correlation; key ambiguity (low = ambiguous)", "r"),
    "mode_majorness": ("Best major-key minus best minor-key correlation (continuous mode)", "r"),
    "key_stability": ("Share of 16-beat windows (HMM-smoothed) in the global key family (same, parallel or relative key)", "proportion"),
    "n_key_changes": ("Number of sustained local-key changes to a different key family (modulations)", "count"),
    "late_upward_modulation": ("1 if the key of the last 30% is 1-2 semitones above the key of the first 60% (same mode)", "0/1"),
    "tuning_offset_cents": ("Deviation of the tuning reference from A4=440 Hz", "cents"),
    "chord_change_rate_per_min": ("Chord changes per minute (harmonic rhythm)", "1/min"),
    "chord_mean_duration_beats": ("Mean chord duration", "beats"),
    "n_unique_chords": ("Number of distinct major/minor triads used", "count"),
    "chord_entropy_bits": ("Entropy of the chord distribution (beat-weighted); harmonic vocabulary diversity", "bits"),
    "top_chord_share": ("Share of beats on the most frequent chord", "proportion"),
    "minor_chord_share": ("Share of chord beats that are minor triads", "proportion"),
    "root_motion_fifth_share": ("Share of chord changes whose roots move by a 4th/5th", "proportion"),
    "harmonic_triadicity": ("Mean cosine fit of beat chroma to the best triad template; low = extended/ambiguous harmony", "0-1"),
    "harmonic_change_rate": ("Mean tonal-centroid (tonnetz) change between beats", "a.u."),
    "roughness_mean": ("Mean sensory roughness/dissonance of spectral peaks (Plomp-Levelt/Sethares)", "0-~0.25"),
    "roughness_sd": ("SD of frame-level roughness", "a.u."),
}
PATTERNS: list = []


def _profile_bank(name: str) -> np.ndarray:
    maj, mnr = PROFILES[name]
    bank = [np.roll(maj, k) for k in range(12)] + [np.roll(mnr, k) for k in range(12)]
    bank = np.array(bank, dtype=np.float64)
    bank -= bank.mean(axis=1, keepdims=True)
    return bank / np.linalg.norm(bank, axis=1, keepdims=True)


def key_correlations(chroma_vec: np.ndarray, bank: np.ndarray) -> np.ndarray:
    v = chroma_vec.astype(np.float64) - chroma_vec.mean()
    n = np.linalg.norm(v)
    return bank @ (v / n) if n > 0 else np.zeros(24)


def _signature(k: int) -> int:
    """Diatonic collection of a key index (0-11 major, 12-23 minor); relatives share it."""
    return k % 12 if k < 12 else (k % 12 + 3) % 12


def _same_family(a: int, b: int) -> bool:
    """Same key, parallel (same tonic) or relative (same signature) key: not a modulation."""
    return (a % 12 == b % 12) or (_signature(a) == _signature(b))


def _absorb_short_runs(states: np.ndarray, min_len: int) -> np.ndarray:
    """Replace runs shorter than min_len (transitional windows) by the preceding run."""
    s = states.copy()
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(s) != 0) + 1, [len(s)]])
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < min_len and len(bounds) > 2:
            s[a:b] = s[a - 1] if a > 0 else s[b]
    return s


def _key_name(k: int) -> tuple[str, str]:
    return PITCHES[k % 12], ("major" if k < 12 else "minor")


def _chord_templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    T, labels = [], []
    for q, ivs in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
        for r in range(12):
            t = np.zeros(12)
            t[[(r + i) % 12 for i in ivs]] = 1
            T.append(t / np.linalg.norm(t))
            labels.append((r, q))
    return np.array(T), labels


def recognise_chords(chroma_beat: np.ndarray, self_prob: float = 0.6, beta: float = 12.0):
    """Template matching + Viterbi smoothing on beat-synchronous chroma.

    Returns (state index per beat, 24 = no-chord) and the per-beat best template fit.
    """
    import librosa
    T, _ = _chord_templates()
    energy = chroma_beat.sum(axis=0)
    C = chroma_beat / (np.linalg.norm(chroma_beat, axis=0, keepdims=True) + 1e-12)
    sim = T @ C                                       # 24 x beats cosine similarity
    fit = sim.max(axis=0)
    logits = np.vstack([beta * sim, np.full((1, sim.shape[1]), beta * 0.55)])  # N state
    silent = energy < 0.02 * (np.percentile(energy, 95) + 1e-12)
    logits[24, silent] = beta * 2
    prob = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob /= prob.sum(axis=0, keepdims=True)
    trans = librosa.sequence.transition_loop(25, self_prob)
    states = librosa.sequence.viterbi_discriminative(prob, trans)
    return states, fit


def _roughness(S: np.ndarray, freqs: np.ndarray, every: int = 4, n_peaks: int = 24) -> np.ndarray:
    from scipy.signal import find_peaks
    band = (freqs >= 50) & (freqs <= 5000)
    f_b = freqs[band]
    frame_e = S.sum(axis=0)
    thr = frame_e.max() * 10 ** (-40 / 20)
    vals = []
    for j in range(0, S.shape[1], every):
        if frame_e[j] < thr:
            continue
        mag = S[band, j]
        pk, _ = find_peaks(mag, height=mag.max() * 10 ** (-60 / 20))
        if pk.size < 2:
            continue
        pk = pk[np.argsort(mag[pk])[::-1][:n_peaks]]
        f, a = f_b[pk], mag[pk] / mag[pk].max()
        fi, fj = np.meshgrid(f, f, indexing="ij")
        ai, aj = np.meshgrid(a, a, indexing="ij")
        iu = np.triu_indices(len(f), 1)
        fmin = np.minimum(fi, fj)[iu]
        df = np.abs(fi - fj)[iu]
        s = 0.24 / (0.0207 * fmin + 18.96)
        aa = (ai * aj)[iu]
        d = aa * (np.exp(-3.51 * s * df) - np.exp(-5.75 * s * df))
        vals.append(d.sum() / (aa.sum() + 1e-12))
    return np.array(vals)


def extract(track: Track, cfg: dict) -> dict:
    import librosa

    out = {k: NAN for k in FEATURES}
    bank = _profile_bank(cfg.get("key_profile", "krumhansl"))
    chroma = track.chroma
    out["tuning_offset_cents"] = float(track.tuning * 100)
    out["tuning_method"] = track.get("tuning_method", lambda: "librosa")

    out["chroma_method"] = track.get("chroma_method", lambda: "unknown")

    # ---- global key: Essentia KeyExtractor (validated, MIREX) when available, else profiles
    r = key_correlations(chroma.sum(axis=1), bank)
    order = np.argsort(r)[::-1]
    k = int(order[0])
    out["key_method"] = f"profile_{cfg.get('key_profile', 'krumhansl')}"
    if cfg.get("key_method", "auto") in ("auto", "essentia"):
        try:
            import essentia.standard as es
            prof = cfg.get("essentia_key_profile", "bgate")
            ek, escale, _ = es.KeyExtractor(profileType=prof, sampleRate=44100)(
                np.ascontiguousarray(track.y_hi, dtype=np.float32))
            pc = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
                  "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}[ek]
            k = pc + (0 if escale == "major" else 12)
            out["key_method"] = f"essentia_{prof}"
        except ImportError:
            pass
    tonic, mode = _key_name(k)
    second = max(r[j] for j in range(24) if j != k)
    out.update(key_tonic=tonic, key_mode=mode, key_label=f"{tonic} {mode}",
               key_clarity=float(r[k]), key_margin=float(r[k] - second),
               mode_majorness=float(r[:12].max() - r[12:].max()))

    # ---- local keys: windowed key correlations smoothed with an HMM (Viterbi)
    cb_sum = track.beat_sync(chroma, aggregate=np.sum)
    win, hop = int(cfg.get("key_window_beats", 16)), int(cfg.get("key_hop_beats", 4))
    starts = list(range(0, max(cb_sum.shape[1] - win, 0) + 1, hop))
    if len(starts) >= 3:
        Rw = np.array([key_correlations(cb_sum[:, s:s + win].sum(axis=1), bank) for s in starts]).T
        logit = 20.0 * Rw
        prob = np.exp(logit - logit.max(axis=0, keepdims=True))
        prob /= prob.sum(axis=0, keepdims=True)
        trans = librosa.sequence.transition_loop(24, float(cfg.get("key_self_transition", 0.9)))
        local = librosa.sequence.viterbi_discriminative(prob, trans).astype(int)
        local = _absorb_short_runs(local, int(cfg.get("key_min_windows", 3)))
        out["key_stability"] = float(np.mean([_same_family(int(x), k) for x in local]))
        out["n_key_changes"] = int(sum(not _same_family(int(a), int(b)) for a, b in zip(local[:-1], local[1:])))
        track.timeseries["local_key_index"] = local
    # late "truck-driver" modulation: key of the last 30% vs key of the first 60%
    nb = cb_sum.shape[1]
    if nb >= 40:
        k1 = int(np.argmax(key_correlations(cb_sum[:, : int(0.6 * nb)].sum(axis=1), bank)))
        k2 = int(np.argmax(key_correlations(cb_sum[:, int(0.7 * nb):].sum(axis=1), bank)))
        up = (k2 % 12 - k1 % 12) % 12
        out["late_upward_modulation"] = int(((k1 < 12) == (k2 < 12)) and up in (1, 2))

    # ---- chords
    cb = track.beat_sync(chroma, aggregate=np.median)
    if cb.shape[1] >= 8:
        states, fit = recognise_chords(cb, float(cfg.get("chord_self_transition", 0.6)))
        _, labels = _chord_templates()
        voiced = states < 24
        out["harmonic_triadicity"] = fnum(fit[voiced].mean()) if voiced.any() else NAN
        st = states[voiced]
        if st.size >= 4:
            changes = np.flatnonzero(np.diff(states) != 0)
            changes = [c for c in changes if states[c] < 24 and states[c + 1] < 24]
            out["chord_change_rate_per_min"] = len(changes) / (track.duration / 60)
            seg_lens = np.diff(np.concatenate([[0], np.flatnonzero(np.diff(st) != 0) + 1, [len(st)]]))
            out["chord_mean_duration_beats"] = float(seg_lens.mean())
            counts = np.bincount(st, minlength=24)
            out["n_unique_chords"] = int((counts > 0).sum())
            out["chord_entropy_bits"] = entropy_bits(counts)
            out["top_chord_share"] = float(counts.max() / counts.sum())
            out["minor_chord_share"] = float(counts[12:].sum() / counts.sum())
            if changes:
                roots = [(labels[states[c]][0], labels[states[c + 1]][0]) for c in changes]
                out["root_motion_fifth_share"] = float(np.mean([((b - a) % 12) in (5, 7) for a, b in roots]))
            # key-relative symbolic sequence for the corpus surprise model
            tonic_pc = k % 12
            seq, prev = [], None
            for s_ in st:
                root, q = labels[int(s_)]
                sym = f"{(root - tonic_pc) % 12}{'M' if q == 'maj' else 'm'}"
                if sym != prev:
                    seq.append(sym)
                    prev = sym
            track.sequences["chords_key_relative"] = seq
            track.sequences["chords_absolute"] = [f"{PITCHES[labels[int(s)][0]]}{'' if labels[int(s)][1] == 'maj' else 'm'}"
                                                  if s < 24 else "N" for s in states]

        tz = librosa.feature.tonnetz(chroma=cb / (cb.max(axis=0, keepdims=True) + 1e-12))
        out["harmonic_change_rate"] = fnum(np.linalg.norm(np.diff(tz, axis=1), axis=0).mean())

    # ---- roughness on the harmonic STFT
    H = np.abs(track.hpss[0])
    freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
    rough = _roughness(H, freqs, every=int(cfg.get("roughness_frame_step", 4)))
    if rough.size:
        out["roughness_mean"] = float(rough.mean())
        out["roughness_sd"] = float(rough.std())
    return out
