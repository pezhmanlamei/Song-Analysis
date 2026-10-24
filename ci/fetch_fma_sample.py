"""Download the Free Music Archive test songs listed in a manifest and verify their SHA-256.

The audio is not stored in this repository. Each song is fetched from the FMA "full" dataset
on Hugging Face (benjamin-paine/free-music-archive-full) using its shard/row position, written
to <manifest folder>/audio/<track_id>.mp3, and checked against the recorded checksum.

    python ci/fetch_fma_sample.py --manifest ci/fma_sample/manifest.csv            # all songs
    python ci/fetch_fma_sample.py --manifest ci/fma_sample/manifest.csv --shard 3/20

--shard K/N selects the same songs as `python -m songfeatures run --shard K/N`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

URL = ("https://huggingface.co/datasets/benjamin-paine/free-music-archive-full/resolve/main/"
       "data/train-{:05d}-of-00972.parquet")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, tries: int = 4):
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            return
        except Exception as e:  # network hiccup: retry with backoff
            if attempt == tries:
                raise
            print(f"  retry {attempt} after error: {e}", flush=True)
            time.sleep(10 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard", help="K/N, same selection as songfeatures run --shard")
    a = ap.parse_args()

    man_path = Path(a.manifest).resolve()
    m = pd.read_csv(man_path)
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        m = m.iloc[k - 1::n]
    base = man_path.parent
    todo = []
    for r in m.itertuples():
        dest = base / r.audio_path
        if dest.exists() and sha256(dest) == r.sha256:
            continue
        todo.append(r)
    print(f"{len(m)} songs selected, {len(m) - len(todo)} already present, {len(todo)} to download")
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for shard, rows in pd.DataFrame(todo).groupby("fma_shard") if todo else []:
            part = Path(tmp) / f"{shard:05d}.parquet"
            t0 = time.time()
            download(URL.format(int(shard)), part)
            audio = pq.read_table(part, columns=["audio"]).column("audio")
            for r in rows.itertuples():
                item = audio[int(r.fma_row)].as_py()
                dest = base / r.audio_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(item["bytes"])
                ok = sha256(dest) == r.sha256
                print(f"  {r.track_id}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
                if not ok:
                    bad.append(r.track_id)
            part.unlink()
            print(f"shard {shard}: {len(rows)} songs in {time.time() - t0:.0f}s", flush=True)
    if bad:
        sys.exit(f"checksum mismatch for {bad}")
    print("all songs present and verified")


if __name__ == "__main__":
    main()
