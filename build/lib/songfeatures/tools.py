"""Dataset helpers.

prepare_manifest  match a track list (CSV/XLSX with Spotify track_id, artists, track_name,
                  album_name, ...) to audio files and write the pipeline manifest.
compare_spotify   compare open_* features with Spotify's audio features (validation table).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import AUDIO_EXT

SPOTIFY_ID = re.compile(r"(?<![0-9A-Za-z])([0-9A-Za-z]{22})(?![0-9A-Za-z])")
COLUMN_ALIASES = {"artists": "artists", "artist": "artists", "artist_name": "artists",
                  "track_name": "title", "title": "title", "name": "title", "song": "title",
                  "album_name": "album", "album": "album", "track_id": "track_id", "id": "track_id",
                  "spotify_id": "track_id", "release_date": "release_date", "year": "year",
                  "track_genre": "genre", "genre": "genre"}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")


def norm(text) -> str:
    """Lower-case, strip accents, drop bracketed parts and version tags, keep letters/digits."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)
    t = re.sub(r"\s-\s.*(remaster|version|edit|mix|live|acoustic|mono|stereo).*$", " ", t)
    t = re.sub(r"\b(feat|ft|featuring)\.?\s.*$", " ", t)
    t = t.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _first_artist(a) -> str:
    return re.split(r"[;,/]| & | feat\.? | ft\.? ", a)[0] if isinstance(a, str) else ""


def _file_tags(p: Path) -> tuple[str, str]:
    try:
        import mutagen
        f = mutagen.File(str(p), easy=True)
        if f and f.tags:
            return (f.tags.get("artist", [""])[0], f.tags.get("title", [""])[0])
    except Exception:
        pass
    return "", ""


def prepare_manifest(dataset: str, audio_dir: str | None, out: str, dedupe: str = "track_id",
                     pilot: int = 0, seed: int = 1) -> dict:
    df = read_table(dataset).rename(columns=lambda c: COLUMN_ALIASES.get(str(c).lower(), c))
    if "track_id" not in df:
        raise ValueError("dataset needs a track_id (Spotify ID) column")
    n_rows = len(df)
    if "genre" in df:   # keep all genre labels of a track that is listed once per genre
        genres = df.groupby("track_id")["genre"].apply(lambda s: "|".join(sorted(set(map(str, s)))))
    df = df.drop_duplicates("track_id").copy()
    if "genre" in df:
        df["genres_all"] = df["track_id"].map(genres)
    n_unique = len(df)
    if dedupe == "artist_title" and {"artists", "title"} <= set(df.columns):
        key = df["artists"].map(norm) + "|" + df["title"].map(norm)
        df["n_versions"] = key.map(key.value_counts())
        df = df[~key.duplicated()].copy()
    df["artist"] = df["artists"].map(_first_artist) if "artists" in df else ""
    df["audio_path"] = None
    if audio_dir:
        df["match_method"] = None
    report = {"dataset_rows": n_rows, "unique_track_ids": n_unique, "after_dedupe": len(df)}

    if audio_dir:
        files = [p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT]
        report["audio_files"] = len(files)
        by_id = {}
        for p in files:
            for m in SPOTIFY_ID.findall(str(p.relative_to(audio_dir))):
                by_id.setdefault(m, p)
        # name index: "artist title" and "title artist" from file names and (optional) tags
        by_name: dict[str, list[Path]] = {}
        for p in files:
            stem = norm(p.stem.replace("_", " "))
            keys = {stem}
            ta, tt = _file_tags(p)
            if ta and tt:
                keys |= {f"{norm(ta)} {norm(tt)}", f"{norm(tt)} {norm(ta)}"}
            for k in keys:
                by_name.setdefault(re.sub(r"^\d+\s+", "", k), []).append(p)   # drop track numbers
        used = set()
        for i, r in df.iterrows():
            p = by_id.get(r["track_id"])
            method = "spotify_id_in_path" if p else None
            if p is None and isinstance(r.get("title"), str):
                a, t = norm(r["artist"]), norm(r["title"])
                cands = by_name.get(f"{a} {t}", []) + by_name.get(f"{t} {a}", [])
                cands = list(dict.fromkeys(c for c in cands if c not in used))
                if len(cands) == 1:
                    p, method = cands[0], "artist_title"
                elif len(cands) > 1:
                    method = "ambiguous"
            if p is not None:
                used.add(p)
                df.at[i, "audio_path"] = str(p.absolute())
            df.at[i, "match_method"] = method or "unmatched"
        report["matched"] = int(df["audio_path"].notna().sum())
        report["by_method"] = df["match_method"].value_counts().to_dict()
        report["unused_audio_files"] = len(files) - len(used)
        outp = Path(out)
        df[df["audio_path"].isna()].to_csv(outp.with_name(outp.stem + "_unmatched.csv"), index=False)
        pd.DataFrame({"audio_path": [str(p) for p in files if p not in used]}).to_csv(
            outp.with_name(outp.stem + "_unused_files.csv"), index=False)
        df = df[df["audio_path"].notna()]

    if pilot:
        df = df.sample(n=min(pilot, len(df)), random_state=seed)
    lead = ["track_id", "audio_path", "artist", "artists", "title", "album", "release_date", "year", "genre", "genres_all"]
    cols = [c for c in lead if c in df] + [c for c in df.columns if c not in lead]
    df[cols].to_csv(out, index=False)
    report["manifest_rows"] = len(df)
    report["manifest"] = str(out)
    return report


# ------------------------------------------------------------------------------ validation
CONTINUOUS = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness",
              "liveness", "loudness", "tempo", "duration_ms"]
CATEGORICAL = ["key", "mode", "time_signature"]


def compare_spotify(features_csv: str, spotify_table: str, out: str) -> pd.DataFrame:
    f = pd.read_csv(features_csv)
    s = read_table(spotify_table)
    s = s.rename(columns={c: c.lower() for c in s.columns}).drop_duplicates("track_id")
    m = f.merge(s, on="track_id", how="inner", suffixes=("", "_spotify"))
    rows = []
    for c in CONTINUOUS:
        a, b = f"open_{c}", c if c in s and c not in f else f"{c}_spotify"
        if a not in m or b not in m:
            continue
        x, y = pd.to_numeric(m[a], errors="coerce"), pd.to_numeric(m[b], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        row = {"feature": c, "n": int(ok.sum()), "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
               "spearman_rho": float(x[ok].rank().corr(y[ok].rank())),
               "mean_open": float(x[ok].mean()), "mean_spotify": float(y[ok].mean())}
        if c == "tempo":   # agreement allowing octave (double/half tempo) errors
            ratio = x[ok] / y[ok]
            row["agreement"] = float(np.mean(np.min(np.abs(np.log2(ratio.values[:, None]) - np.array([[-1, 0, 1]])), axis=1) < np.log2(1.04)))
        rows.append(row)
    for c in CATEGORICAL:
        a, b = f"open_{c}", c if c not in f else f"{c}_spotify"
        if a in m and b in m:
            x, y = pd.to_numeric(m[a], errors="coerce"), pd.to_numeric(m[b], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum():
                rows.append({"feature": c, "n": int(ok.sum()), "agreement": float((x[ok] == y[ok]).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    return res
