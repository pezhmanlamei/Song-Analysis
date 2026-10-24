"""Song structure and energy arc (full-song features).

Default backend: Laplacian structural segmentation (McFee & Ellis, 2014) on
beat-synchronous CQT + MFCC, plus a documented chorus heuristic (most-repeated,
loudest section type). Optional backend `allin1` (Kim & Nam, 2023) gives
learned functional labels (intro/verse/chorus/...).
"""
from __future__ import annotations

import numpy as np

from ..audio import SR, Track
from ..utils import NAN, entropy_bits, fnum, linear_slope, runs
from .loudness import windowed_loudness

NAME = "structure"
INFO = {
    "tool": "librosa + scikit-learn (Laplacian segmentation); optional allin1",
    "reference": "McFee & Ellis (2014) ISMIR, Laplacian segmentation; Foote (2000); "
                 "Kim & Nam (2023) All-In-One; Van Balen et al. (2013) chorus features",
}
FEATURES = {
    "structure_backend": ("Segmentation backend used", "text"),
    "structure_reliable": ("0 if segmentation failed (one section for a song > 45 s); section/chorus features are then NaN", "0/1"),
    "n_sections": ("Number of contiguous sections", "count"),
    "n_section_types": ("Number of distinct section types (clusters / labels)", "count"),
    "section_mean_duration_s": ("Mean section duration", "s"),
    "section_duration_cv": ("Coefficient of variation of section durations", "ratio"),
    "structural_entropy_bits": ("Entropy of time spent in each section type; structural complexity", "bits"),
    "repetition_score": ("Mean over beats of max self-similarity to a non-adjacent moment (>16 beats away)", "0-1"),
    "intro_duration_s": ("Start time of the second section (length of the opening section)", "s"),
    "starts_with_chorus": ("1 if the first section is the chorus type (hook-first)", "0/1"),
    "time_to_first_chorus_s": ("Onset of the first chorus section", "s"),
    "chorus_share": ("Share of track duration in chorus sections", "proportion"),
    "n_chorus_repeats": ("Number of chorus sections", "count"),
    "chorus_loudness_contrast_db": ("Mean short-term loudness of chorus minus non-chorus sections", "LU"),
    "energy_slope_lu_per_min": ("Linear trend of short-term loudness over the track", "LU/min"),
    "energy_peak_position": ("Relative position (0-1) of the loudest 5-s passage", "0-1"),
    "energy_arc_curvature": ("Quadratic coefficient of loudness vs normalised time (<0 = rise-then-fall arc)", "LU"),
    "n_drops": ("Sudden loudness increases (>= drop_db after a quieter >= 7 s passage)", "count"),
    "first_drop_time_s": ("Time of the first drop", "s"),
    "n_buildups": ("8-s windows with a gradual monotonic (Spearman >= .8) loudness rise >= 4 LU", "count"),
    "leading_silence_s": ("Silence before the first sound (< -60 LUFS momentary)", "s"),
    "trailing_silence_s": ("Silence after the last sound", "s"),
    "ending_slope_lu_per_s": ("Loudness slope over the final 12 s of sound", "LU/s"),
    "ending_type": ("fade_out (gradual decline >= 5 s while notes continue), decay (final note/chord rings out) "
                    "or cold_cut (stops abruptly while playing)", "category"),
    "ends_with_fade": ("1 if ending_type is fade_out", "0/1"),
    "fade_out_duration_s": ("Length of the fade-out (0 if no fade)", "s"),
    "ending_decline_s": ("Time from the last full-level moment (within 3 LU of the pre-ending level) to the end of sound", "s"),
}
PATTERNS: list = []


# --------------------------------------------------------------------------- segmentation
def laplacian_segments(track: Track, n_types="auto", min_beats: int = 8):
    """Return per-beat section labels using McFee & Ellis (2014) spectral clustering."""
    import librosa
    import scipy.linalg
    import scipy.ndimage
    import scipy.sparse.csgraph
    from sklearn.cluster import KMeans

    frames = track.beats["frames"]
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=track.y, sr=SR, bins_per_octave=36, n_bins=252,
                                                   hop_length=512)), ref=np.max)
    Csync = librosa.util.sync(C, frames, aggregate=np.median)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    Rf = librosa.segment.timelag_filter(scipy.ndimage.median_filter)(R, size=(1, 7))
    Msync = track.beat_sync(track.mfcc)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance) + 1e-12
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    n = min(Rf.shape[0], R_path.shape[0])
    Rf, R_path = Rf[:n, :n], R_path[:n, :n]
    deg_path, deg_rec = R_path.sum(axis=1), Rf.sum(axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / (np.sum((deg_path + deg_rec) ** 2) + 1e-12)
    A = mu * Rf + (1 - mu) * R_path
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    if n_types == "auto":
        gaps = np.diff(evals[:11])
        k = int(np.clip(np.argmax(gaps[2:9]) + 3, 3, 8))   # eigengap heuristic, k in [3, 8]
    else:
        k = int(n_types)
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5
    X = evecs[:, :k] / (Cnorm[:, k - 1:k] + 1e-12)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    labels = _mode_filter(labels, min_beats + 1)
    labels = _merge_short(labels, min_beats)
    return labels


def _mode_filter(labels: np.ndarray, width: int) -> np.ndarray:
    """Majority vote over a sliding window: removes 1-3 beat label flicker before merging."""
    n, h = len(labels), width // 2
    out = labels.copy()
    for i in range(n):
        w = labels[max(0, i - h): i + h + 1]
        vals, counts = np.unique(w, return_counts=True)
        best = vals[counts == counts.max()]
        out[i] = labels[i] if labels[i] in best else best[0]
    return out


def _merge_short(labels: np.ndarray, min_len: int) -> np.ndarray:
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(labels) != 0) + 1, [len(labels)]])
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_len and len(bounds) > 2:
                fill = labels[s - 1] if s > 0 else labels[e]
                labels[s:e] = fill
                changed = True
                break
    return labels


def _allin1_segments(track: Track):
    import allin1
    res = allin1.analyze(str(track.path))
    return [(float(s.start), float(s.end), str(s.label)) for s in res.segments
            if s.label not in ("start", "end")]


def _sections_from_beats(labels, beat_times, duration):
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(labels) != 0) + 1])
    secs = []
    for i, b in enumerate(bounds):
        start = 0.0 if i == 0 else float(beat_times[b])
        end = float(beat_times[bounds[i + 1]]) if i + 1 < len(bounds) else duration
        secs.append((start, end, int(labels[b])))
    return secs


def _mean_loud(wl, s, e):
    m = (wl["t_st"] >= s) & (wl["t_st"] < e)
    return float(np.mean(wl["st"][m])) if m.any() else NAN


# --------------------------------------------------------------------------- energy arc
def _drops_and_buildups(t, L, drop_db):
    from scipy.stats import spearmanr
    step = np.median(np.diff(t))
    drops, last = [], -1e9
    for i in range(len(t)):
        if t[i] < 8 or t[i] - last < 15:
            continue
        pre = L[(t >= t[i] - 8) & (t < t[i] - 1)]
        post = L[(t >= t[i]) & (t < t[i] + 4)]
        if pre.size and post.size and np.median(post) - np.median(pre) >= drop_db:
            drops.append(float(t[i]))
            last = t[i]
    n_build, w = 0, int(8 / step)
    i = 0
    while i + w < len(t):
        seg = L[i:i + w]
        half = w // 2
        gradual = (seg[half] - seg[0] >= 1.5) and (seg[-1] - seg[half] >= 1.5)  # not a single step
        if seg[-1] - seg[0] >= 4 and gradual and spearmanr(np.arange(w), seg)[0] >= 0.8:
            n_build += 1
            i += w
        else:
            i += max(1, w // 4)
    return drops, n_build


def classify_ending(track: Track, t, Ls, end_t: float, fade_min_s: float = 5.0) -> dict:
    """fade_out: level declines gradually for >= fade_min_s while notes keep sounding;
    cold_cut: music stops abruptly at full level; decay: final note/chord rings out."""
    import librosa
    ref_m = (t >= end_t - 30) & (t < end_t - 12)
    if ref_m.sum() < 10:
        ref_m = t < end_t - 12
    ref = float(np.median(Ls[ref_m])) if ref_m.any() else float(np.max(Ls))
    held = np.flatnonzero((t <= end_t) & (Ls >= ref - 3))
    t_start = float(t[held[-1]]) if held.size else end_t - 12
    decline = max(end_t - t_start, 0.0)
    on = librosa.onset.onset_detect(onset_envelope=track.onset_env, sr=SR, hop_length=512, units="time")
    rate = float(((on >= t_start) & (on <= end_t)).sum() / max(decline, 0.5))
    last_onset_gap = float(end_t - on[on <= end_t].max()) if (on <= end_t).any() else NAN
    if decline >= fade_min_s and rate >= 1.0:
        kind = "fade_out"
    elif decline < 1.5 and np.isfinite(last_onset_gap) and last_onset_gap < 1.0:
        kind = "cold_cut"
    else:
        kind = "decay"
    return {"ending_type": kind, "ends_with_fade": int(kind == "fade_out"),
            "fade_out_duration_s": decline if kind == "fade_out" else 0.0,
            "ending_decline_s": decline}


def extract(track: Track, cfg: dict) -> dict:
    out = {k: NAN for k in FEATURES}
    dur = track.duration
    wl = windowed_loudness(track)
    bt = track.beats["times"]

    # ---- sections
    backend = cfg.get("backend", "laplacian")
    secs = None
    if backend == "allin1":
        try:
            secs = _allin1_segments(track)
            chorus_type = "chorus"
        except ImportError:
            backend = "laplacian"
    if secs is None:
        labels = laplacian_segments(track, cfg.get("n_section_types", "auto"), int(cfg.get("min_section_beats", 8)))
        secs = _sections_from_beats(labels, bt, dur)
        types = sorted({s[2] for s in secs})
        # chorus heuristic: among types occurring >= 2 times, the loudest (tie -> most frequent)
        cand = []
        for ty in types:
            occ = [s for s in secs if s[2] == ty]
            if len(occ) >= 2:
                loud = np.nanmean([_mean_loud(wl, s, e) for s, e, _ in occ])
                cand.append((loud, len(occ), ty))
        chorus_type = max(cand, key=lambda c: (c[0] if np.isfinite(c[0]) else -np.inf, c[1]))[2] if cand else None
    out["structure_backend"] = backend

    # a song longer than 45 s that yields a single section means segmentation failed
    reliable = len(secs) >= 2 or dur < 45
    out["structure_reliable"] = int(reliable)
    if not reliable:
        secs, chorus_type = [], None
    durs = np.array([e - s for s, e, _ in secs]) if secs else np.array([np.nan])
    tdur = {}
    for s, e, ty in secs:
        tdur[ty] = tdur.get(ty, 0) + (e - s)
    out["n_sections"] = len(secs) if secs else NAN
    out["n_section_types"] = len(tdur) if secs else NAN
    out["section_mean_duration_s"] = float(durs.mean())
    out["section_duration_cv"] = float(durs.std() / durs.mean()) if secs and durs.mean() > 0 else NAN
    out["structural_entropy_bits"] = entropy_bits(tdur.values()) if secs else NAN
    out["intro_duration_s"] = (float(secs[1][0]) if len(secs) > 1 else float(dur)) if secs else NAN
    if chorus_type is not None:
        ch = [(s, e) for s, e, ty in secs if ty == chorus_type]
        out["starts_with_chorus"] = int(secs[0][2] == chorus_type)
        out["time_to_first_chorus_s"] = float(ch[0][0])
        out["chorus_share"] = float(sum(e - s for s, e in ch) / dur)
        out["n_chorus_repeats"] = len(ch)
        non = [(s, e) for s, e, ty in secs if ty != chorus_type]
        if non:
            out["chorus_loudness_contrast_db"] = float(
                np.nanmean([_mean_loud(wl, s, e) for s, e in ch]) - np.nanmean([_mean_loud(wl, s, e) for s, e in non]))
    track.sequences["sections"] = [[round(s, 2), round(e, 2), str(ty)] for s, e, ty in secs]
    track.sequences["chorus_type"] = str(chorus_type)

    # ---- repetition (self-similarity of beat-synchronous chroma + timbre)
    cb = track.beat_sync(track.chroma)
    cb = cb / (np.linalg.norm(cb, axis=0, keepdims=True) + 1e-12)
    mb = track.beat_sync(track.mfcc[1:14])
    mb = (mb - mb.mean(axis=1, keepdims=True)) / (mb.std(axis=1, keepdims=True) + 1e-9)
    mb = mb / (np.linalg.norm(mb, axis=0, keepdims=True) + 1e-12)
    F = np.vstack([cb, mb]) / np.sqrt(2)
    Sm = F.T @ F
    n = Sm.shape[0]
    if n > 40:
        ii, jj = np.indices(Sm.shape)
        Sm[np.abs(ii - jj) <= 16] = -np.inf
        out["repetition_score"] = float(np.clip(np.max(Sm, axis=1), 0, 1).mean())

    # ---- energy arc on momentary loudness smoothed to 1 s
    t, Lm = wl["t_mom"], wl["mom"]
    sound = Lm > -60
    if sound.any():
        first, last = np.flatnonzero(sound)[[0, -1]]
        out["leading_silence_s"] = float(max(t[first] - 0.2, 0.0))
        out["trailing_silence_s"] = float(max(dur - t[last] - 0.2, 0.0))
        from scipy.ndimage import uniform_filter1d
        Ls = uniform_filter1d(np.where(sound, Lm, -60.0), size=10, mode="nearest")
        st = wl["st"]
        keep = st > -70
        out["energy_slope_lu_per_min"] = linear_slope(wl["t_st"][keep] / 60, st[keep])
        L5 = uniform_filter1d(Ls, size=50, mode="nearest")
        out["energy_peak_position"] = float(t[int(np.argmax(L5))] / dur)
        tn = wl["t_st"][keep] / dur
        if keep.sum() > 10:
            out["energy_arc_curvature"] = float(np.polyfit(tn, st[keep], 2)[0])
        drops, nb = _drops_and_buildups(t, Ls, float(cfg.get("drop_db", 6.0)))
        out["n_drops"] = len(drops)
        out["first_drop_time_s"] = drops[0] if drops else NAN
        out["n_buildups"] = nb
        end_t = t[last]
        m = (t >= end_t - 12) & (t <= end_t)
        if m.sum() > 20:
            out["ending_slope_lu_per_s"] = linear_slope(t[m], Ls[m])
            out.update(classify_ending(track, t, Ls, end_t, float(cfg.get("fade_min_s", 5.0))))
    return out
