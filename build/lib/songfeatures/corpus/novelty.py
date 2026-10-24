"""Novelty and typicality in embedding space (Discogs-EffNet and/or CLAP embeddings).

* typicality_corpus: mean cosine similarity to every other song in the corpus.
* typicality_period / novelty_period: similarity to songs released within +/- window years
  (Askin & Mauskapf, 2017 operationalise typicality relative to contemporaries).
* knn_distance: mean cosine distance to the k most similar songs (local uniqueness).
* typicality_genre: mean similarity to songs of the same genre (manifest `genre`, or the
  Essentia parent genre when not provided).
* artist_novelty: cosine distance to the centroid of the same artist's earlier songs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "novelty"
INFO = {
    "tool": "numpy on L2-normalised song embeddings (Discogs-EffNet mean, CLAP mean)",
    "reference": "Askin & Mauskapf (2017) What makes popular culture popular? ASR; "
                 "Berger & Packard (2018) atypical songs, Psychological Science",
}
FEATURES = {}
PATTERNS = [
    (r"^typicality_corpus_", "Mean cosine similarity to all other songs ({0} embedding)", "-1..1"),
    (r"^typicality_period_", "Mean cosine similarity to songs released within +/- window years ({0})", "-1..1"),
    (r"^novelty_period_", "1 - typicality_period ({0})", "0..2"),
    (r"^n_period_peers_", "Number of contemporaneous songs used for the period comparison", "count"),
    (r"^knn_distance_", "Mean cosine distance to the k nearest songs in the corpus ({0})", "0..2"),
    (r"^typicality_genre_", "Mean cosine similarity to same-genre songs ({0})", "-1..1"),
    (r"^artist_novelty_", "Cosine distance to the centroid of the artist's earlier songs ({0})", "0..2"),
]


def _year(meta: pd.DataFrame) -> pd.Series:
    if "year" in meta:
        return pd.to_numeric(meta["year"], errors="coerce")
    if "release_date" in meta:
        return pd.to_datetime(meta["release_date"], errors="coerce").dt.year
    return pd.Series(np.nan, index=meta.index)


def corpus_novelty(emb: dict[str, dict[str, np.ndarray]], meta: pd.DataFrame, cfg: dict) -> dict[str, dict]:
    """emb: track_id -> {embedding_name: vector}; meta indexed by track_id."""
    results: dict[str, dict] = {t: {} for t in emb}
    names = sorted({k for d in emb.values() for k in d})
    window = float(cfg.get("period_window_years", 1))
    k_nn = int(cfg.get("knn", 10))
    for name in names:
        ids = [t for t in emb if name in emb[t]]
        if len(ids) < int(cfg.get("min_corpus_songs", 20)):
            continue
        X = np.stack([emb[t][name] for t in ids]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        tag = name.replace("_mean", "")
        m = meta.reindex(ids)
        yr = _year(m).to_numpy()
        genre = m["genre"].astype(str).to_numpy() if "genre" in m else m.get("genre_parent_top1", pd.Series([None] * len(ids))).astype(str).to_numpy()
        artist = m["artist"].astype(str).to_numpy() if "artist" in m else None
        date = pd.to_datetime(m["release_date"], errors="coerce").to_numpy() if "release_date" in m else None
        for i, t in enumerate(ids):
            row = (X @ X[i]).astype(np.float64)       # one row at a time: O(n) memory
            row[i] = np.nan
            r = results[t]
            r[f"typicality_corpus_{tag}"] = float(np.nanmean(row))
            sims = np.sort(row[~np.isnan(row)])[::-1]
            r[f"knn_distance_{tag}"] = float(1 - sims[:k_nn].mean())
            if np.isfinite(yr[i]):
                peers = np.isfinite(yr) & (np.abs(yr - yr[i]) <= window)
                peers[i] = False
                r[f"n_period_peers_{tag}"] = int(peers.sum())
                if peers.sum() >= 5:
                    tp = float(np.nanmean(row[peers]))
                    r[f"typicality_period_{tag}"] = tp
                    r[f"novelty_period_{tag}"] = 1 - tp
            if genre is not None and genre[i] not in ("nan", "None", ""):
                same = genre == genre[i]
                same[i] = False
                if same.sum() >= 3:
                    r[f"typicality_genre_{tag}"] = float(np.nanmean(row[same]))
            if artist is not None and artist[i] not in ("nan", "None", ""):
                when = date if date is not None else yr
                if when is not None and not pd.isna(when[i]):
                    earlier = (artist == artist[i]) & np.array([not pd.isna(w) and w < when[i] for w in when])
                    if earlier.any():
                        c = X[earlier].mean(axis=0)
                        c /= np.linalg.norm(c) + 1e-12
                        r[f"artist_novelty_{tag}"] = float(1 - X[i] @ c)
    return results
