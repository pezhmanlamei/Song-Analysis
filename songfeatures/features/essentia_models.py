"""Pretrained Essentia models (MTG-UPF): emotion over time (valence/arousal), moods,
mood/theme tags, instruments, genres (Discogs 400), danceability, voice/instrumental,
approachability and engagement. Also stores Discogs-EffNet / MusiCNN embeddings for the
corpus-level novelty stage.

Licence note: the Essentia model weights are CC BY-NC-SA 4.0 (non-commercial research use).
Run `python -m songfeatures download-models` once to fetch them (~300 MB).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..audio import SR_HI, Track
from ..utils import NAN, series_summary, slug

NAME = "essentia_models"
INFO = {
    "tool": "essentia-tensorflow with MTG pretrained models (Discogs-EffNet, MSD-MusiCNN heads)",
    "reference": "Alonso-Jimenez, Serra & Bogdanov (2023) Efficient supervised training of audio transformers "
                 "for music representation learning (Discogs-EffNet), ISMIR; Pons & Serra (2019) MusiCNN; "
                 "Aljanaki, Yang & Soleymani (2017) DEAM; Bogdanov et al. (2019) MTG-Jamendo",
}
BASE = "https://essentia.upf.edu/models/"
EMBEDDERS = {
    "effnet": dict(file="discogs-effnet-bs64-1.pb", path="feature-extractors/discogs-effnet/",
                   algo="TensorflowPredictEffnetDiscogs", output="PartitionedCall:1"),
    "musicnn": dict(file="msd-musicnn-1.pb", path="feature-extractors/musicnn/",
                    algo="TensorflowPredictMusiCNN", output="model/dense/BiasAdd"),
}
# kind: binary (positive class name) | multilabel (prefix) | regression (column names) | genre
HEADS = [
    dict(name="mood_happy", emb="effnet", dir="mood_happy", file="mood_happy-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="happy"),
    dict(name="mood_sad", emb="effnet", dir="mood_sad", file="mood_sad-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="sad"),
    dict(name="mood_aggressive", emb="effnet", dir="mood_aggressive", file="mood_aggressive-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="aggressive"),
    dict(name="mood_relaxed", emb="effnet", dir="mood_relaxed", file="mood_relaxed-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="relaxed"),
    dict(name="mood_party", emb="effnet", dir="mood_party", file="mood_party-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="party"),
    dict(name="mood_acoustic", emb="effnet", dir="mood_acoustic", file="mood_acoustic-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="acoustic"),
    dict(name="mood_electronic", emb="effnet", dir="mood_electronic", file="mood_electronic-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="electronic"),
    dict(name="danceability_model", emb="effnet", dir="danceability", file="danceability-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="danceable"),
    dict(name="voice_prob", emb="effnet", dir="voice_instrumental", file="voice_instrumental-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="voice"),
    dict(name="tonal_prob", emb="effnet", dir="tonal_atonal", file="tonal_atonal-discogs-effnet-1.pb", output="model/Softmax", kind="binary", positive="tonal"),
    dict(name="approachability", emb="effnet", dir="approachability", file="approachability_regression-discogs-effnet-1.pb", output="model/Identity", kind="regression", columns=["approachability"]),
    dict(name="engagement", emb="effnet", dir="engagement", file="engagement_regression-discogs-effnet-1.pb", output="model/Identity", kind="regression", columns=["engagement"]),
    dict(name="moodtheme", emb="effnet", dir="mtg_jamendo_moodtheme", file="mtg_jamendo_moodtheme-discogs-effnet-1.pb", output="model/Sigmoid", kind="multilabel"),
    dict(name="instrument", emb="effnet", dir="mtg_jamendo_instrument", file="mtg_jamendo_instrument-discogs-effnet-1.pb", output="model/Sigmoid", kind="multilabel"),
    dict(name="genre", emb="effnet", dir="genre_discogs400", file="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0", kind="genre"),
    dict(name="deam", emb="musicnn", dir="deam", file="deam-msd-musicnn-2.pb", output="model/Identity", kind="emotion"),
    dict(name="emomusic", emb="musicnn", dir="emomusic", file="emomusic-msd-musicnn-2.pb", output="model/Identity", kind="emotion"),
]

FEATURES = {
    "essentia_missing_heads": ("Model heads requested but not found on disk (skipped)", "text"),
    "approachability":("Model-predicted approachability (mainstream vs niche), MTG regression head", "0-1"),
    "engagement": ("Model-predicted engagement (background vs active listening), MTG regression head", "0-1"),
    "genre_top1": ("Most probable Discogs-400 style (Parent---Style)", "text"),
    "genre_top1_prob": ("Probability of genre_top1", "0-1"),
    "genre_top3": ("Top three Discogs-400 styles, '|' separated", "text"),
    "genre_parent_top1": ("Most probable Discogs parent genre", "text"),
}
for h in HEADS:
    if h["kind"] == "binary":
        FEATURES[h["name"]] = (f"Mean model probability of '{h['positive']}' ({h['dir']} head, Discogs-EffNet)", "0-1")
PATTERNS = [
    (r"^(deam|emomusic)_(valence|arousal)_", "Valence/arousal (1-9 scale) predicted every ~1.5 s by the {0} model; time-series summary", "1-9"),
    (r"^moodtheme_", "Mean probability of an MTG-Jamendo mood/theme tag (56 tags)", "0-1"),
    (r"^instrument_", "Mean probability of an MTG-Jamendo instrument tag (40 tags)", "0-1"),
    (r"^genre_parent_", "Max probability over the Discogs styles of this parent genre (15 parents)", "0-1"),
]

_ALGOS: dict = {}


def model_dir(cfg) -> Path:
    return Path(cfg.get("essentia_model_dir", "models/essentia")).expanduser()


def download(cfg: dict, heads: list[str] | None = None):
    """Download embedding extractors and the selected classifier heads (+ .json metadata)."""
    import urllib.request
    d = model_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    wanted = [h for h in HEADS if heads is None or h["name"] in heads]
    files = [(e["path"], e["file"]) for e in EMBEDDERS.values()]
    files += [(f"classification-heads/{h['dir']}/", h["file"]) for h in wanted]
    failed = []
    for sub, f in files:
        for name in (f, f.replace(".pb", ".json")):
            target = d / name
            if target.exists():
                continue
            url = BASE + sub + name
            print("downloading", url)
            try:
                _download(url, target)
            except Exception as e:  # a head may have been renamed upstream
                failed.append(url)
                print(f"  WARNING: could not download ({e}); this head will be skipped. "
                      "Check https://essentia.upf.edu/models.html and update HEADS if renamed.")
    return d, failed


def _download(url: str, target: Path, timeout: float = 60, tries: int = 3):
    """Download with a socket timeout and retries; write to .part first so an interrupted
    download is never mistaken for a complete model file."""
    import shutil
    import time
    import urllib.request
    part = target.with_name(target.name + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            part.replace(target)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _meta(path: Path) -> dict:
    j = path.with_suffix(".json")
    return json.loads(j.read_text()) if j.exists() else {}


def _io_names(meta: dict, purpose: str, default_in, default_out):
    """Read input/output node names from the model's .json schema when present."""
    sch = meta.get("schema", {})
    inp = sch.get("inputs", [{}])[0].get("name", default_in) if sch.get("inputs") else default_in
    outs = [o for o in sch.get("outputs", []) if o.get("output_purpose") == purpose]
    return inp, (outs[0]["name"] if outs else default_out)


def _embedder(key: str, cfg):
    import essentia.standard as es
    if key not in _ALGOS:
        e = EMBEDDERS[key]
        p = model_dir(cfg) / e["file"]
        _, out = _io_names(_meta(p), "embeddings", None, e["output"])
        _ALGOS[key] = getattr(es, e["algo"])(graphFilename=str(p), output=out)
    return _ALGOS[key]


def _head(h: dict, cfg):
    import essentia.standard as es
    if h["name"] not in _ALGOS:
        p = model_dir(cfg) / h["file"]
        inp, out = _io_names(_meta(p), "predictions", h.get("input", "model/Placeholder"), h["output"])
        _ALGOS[h["name"]] = (es.TensorflowPredict2D(graphFilename=str(p), input=inp, output=out),
                             _meta(p).get("classes", []))
    return _ALGOS[h["name"]]


def summarise_genre(P: np.ndarray, classes: list[str]) -> dict:
    out = {}
    m = P.mean(axis=0)
    order = np.argsort(m)[::-1]
    out["genre_top1"] = classes[order[0]] if classes else str(order[0])
    out["genre_top1_prob"] = float(m[order[0]])
    out["genre_top3"] = "|".join(classes[i] for i in order[:3]) if classes else ""
    parents: dict[str, float] = {}
    for c, v in zip(classes, m):
        par = c.split("---")[0]
        parents[par] = max(parents.get(par, 0.0), float(v))
    for par, v in parents.items():
        out[f"genre_parent_{slug(par)}"] = v
    if parents:
        out["genre_parent_top1"] = max(parents, key=parents.get)
    return out


def _audio16(track: Track) -> np.ndarray:
    import librosa
    return track.get("audio16", lambda: np.ascontiguousarray(
        librosa.resample(track.y_hi, orig_sr=SR_HI, target_sr=16000), dtype=np.float32))


def effnet_embeddings(track: Track, cfg: dict) -> np.ndarray:
    """Discogs-EffNet patch embeddings of the mix (cached on the track; shared with vocals)."""
    return track.get("emb_effnet", lambda: np.asarray(_embedder("effnet", cfg)(_audio16(track))))


def extract(track: Track, cfg: dict) -> dict:
    out: dict = {}
    audio16 = _audio16(track)
    enabled = cfg.get("heads") or [h["name"] for h in HEADS]
    present = [h["name"] for h in HEADS if (model_dir(cfg) / h["file"]).exists()]
    missing = [n for n in enabled if n not in present]
    if not present:
        raise FileNotFoundError(f"no Essentia models in {model_dir(cfg)}; run: python -m songfeatures download-models")
    enabled = [n for n in enabled if n in present]
    out["essentia_missing_heads"] = "|".join(missing)
    emb = {}
    for key in {h["emb"] for h in HEADS if h["name"] in enabled}:
        emb[key] = effnet_embeddings(track, cfg) if key == "effnet" else np.asarray(_embedder(key, cfg)(audio16))
        track.embeddings[f"{key}_mean"] = emb[key].mean(axis=0)
    for h in HEADS:
        if h["name"] not in enabled:
            continue
        algo, classes = _head(h, cfg)
        P = np.asarray(algo(emb[h["emb"]].astype(np.float32)))
        if h["kind"] == "binary":
            idx = classes.index(h["positive"]) if h["positive"] in classes else 0
            out[h["name"]] = float(P[:, idx].mean())
        elif h["kind"] == "regression":
            out[h["columns"][0]] = float(P[:, 0].mean())
        elif h["kind"] == "multilabel":
            labels = classes or [str(i) for i in range(P.shape[1])]
            for lab, v in zip(labels, P.mean(axis=0)):
                out[f"{h['name']}_{slug(lab)}"] = float(v)
        elif h["kind"] == "genre":
            out.update(summarise_genre(P, classes))
        elif h["kind"] == "emotion":
            t = np.linspace(0, track.duration, num=P.shape[0], endpoint=False) + track.duration / P.shape[0] / 2
            out.update(series_summary(f"{h['name']}_valence", t, P[:, 0], track.duration))
            out.update(series_summary(f"{h['name']}_arousal", t, P[:, 1], track.duration))
            track.timeseries[f"{h['name']}_valence"] = P[:, 0].astype(np.float32)
            track.timeseries[f"{h['name']}_arousal"] = P[:, 1].astype(np.float32)
            track.timeseries[f"{h['name']}_t"] = t.astype(np.float32)
    return out
