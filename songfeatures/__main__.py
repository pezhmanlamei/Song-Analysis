"""Command line interface.

  python -m songfeatures run --manifest songs.csv --out results --config config.yaml --workers 4
  python -m songfeatures run --audio-dir /data/songs --out results --modules loudness,rhythm
  python -m songfeatures download-models --config config.yaml
  python -m songfeatures check
  python -m songfeatures codebook --out codebook.csv
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="songfeatures", description="Open-source audio feature pipeline for full songs")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="extract features")
    r.add_argument("--manifest", help="CSV with audio_path (+ optional track_id, artist, title, release_date/year, "
                                      "genre, lyrics_path, vocals_path, ...)")
    r.add_argument("--audio-dir", help="folder scanned recursively for audio files (alternative to --manifest)")
    r.add_argument("--out", required=True, help="output folder")
    r.add_argument("--config", help="YAML config (defaults are used for anything not set)")
    r.add_argument("--workers", type=int, default=1, help="parallel processes for the signal-processing pass")
    r.add_argument("--modules", help="comma-separated subset of modules to run")
    r.add_argument("--limit", type=int, help="only the first N tracks (pilot runs)")
    r.add_argument("--force", action="store_true", help="recompute even if cached")
    r.add_argument("--shard", help="K/N: process every N-th track starting at K (1-based), e.g. a SLURM array "
                                   "task; skips the corpus stage and merging - run `merge` afterwards")

    mg = sub.add_parser("merge", help="after sharded runs: corpus stage + merged features.csv/codebook/errors")
    mg.add_argument("--manifest", required=True)
    mg.add_argument("--out", required=True)
    mg.add_argument("--config")

    d = sub.add_parser("download-models", help="download Essentia models and pre-fetch Demucs/Whisper/CLAP weights")
    d.add_argument("--config")
    d.add_argument("--essentia-only", action="store_true")

    pm = sub.add_parser("prepare-manifest", help="match a track list (CSV/XLSX) to audio files and write a manifest")
    pm.add_argument("--dataset", required=True, help="CSV/XLSX with track_id (+ artists, track_name, album_name, ...)")
    pm.add_argument("--audio-dir", help="folder with the audio files (matched by Spotify ID in the path, "
                                        "else by artist + title in file name / tags)")
    pm.add_argument("--out", default="manifest.csv")
    pm.add_argument("--dedupe", choices=["track_id", "artist_title"], default="track_id",
                    help="track_id: one row per Spotify ID; artist_title: also collapse versions of the same song")
    pm.add_argument("--pilot", type=int, default=0, help="random sample of N matched songs")

    cs = sub.add_parser("compare-spotify", help="validate open_* features against Spotify's audio features")
    cs.add_argument("--features", required=True, help="features.csv from a run")
    cs.add_argument("--spotify", required=True, help="CSV/XLSX with track_id + Spotify audio-feature columns")
    cs.add_argument("--out", default="spotify_comparison.csv")

    sub.add_parser("check", help="report which optional components are installed")

    c = sub.add_parser("codebook", help="write the full feature codebook")
    c.add_argument("--out", default="codebook.csv")

    a = p.parse_args(argv)
    from . import pipeline

    if a.cmd == "run":
        cfg = pipeline.load_config(a.config)
        df = pipeline.load_manifest(a.manifest, a.audio_dir)
        if a.limit:
            df = df.head(a.limit)
        mods = [m.strip() for m in a.modules.split(",")] if a.modules else None
        finalize = True
        if a.shard:
            k, n = (int(x) for x in a.shard.split("/"))
            if not 1 <= k <= n:
                raise SystemExit("--shard must be K/N with 1 <= K <= N")
            df = df.iloc[k - 1::n]
            finalize = n == 1
            mods = [m for m in (mods or pipeline.CORE + pipeline.MODEL) if m not in pipeline.CORPUS]
        pipeline.run(df, a.out, cfg, workers=a.workers, modules=mods, force=a.force, finalize=finalize)

    elif a.cmd == "merge":
        cfg = pipeline.load_config(a.config)
        df = pipeline.load_manifest(a.manifest, None)
        pipeline.run(df, a.out, cfg, modules=list(pipeline.CORPUS))

    elif a.cmd == "download-models":
        cfg = pipeline.load_config(a.config)
        from .features import essentia_models
        d, failed = essentia_models.download(pipeline.module_cfg(cfg, "essentia_models"),
                                             cfg["essentia_models"].get("heads"))
        print("Essentia models in", d, "| failed:", len(failed))
        if not a.essentia_only:
            _prefetch(cfg)

    elif a.cmd == "prepare-manifest":
        from .tools import prepare_manifest
        rep = prepare_manifest(a.dataset, a.audio_dir, a.out, a.dedupe, a.pilot)
        for k, v in rep.items():
            print(f"  {k}: {v}")

    elif a.cmd == "compare-spotify":
        from .tools import compare_spotify
        print(compare_spotify(a.features, a.spotify, a.out).round(3).to_string(index=False))

    elif a.cmd == "check":
        _check()

    elif a.cmd == "codebook":
        from .codebook import build_codebook
        build_codebook().to_csv(a.out, index=False)
        print("wrote", a.out)


def _prefetch(cfg):
    steps = [
        ("Demucs", lambda: __import__("demucs.pretrained", fromlist=["get_model"]).get_model(cfg["vocals"]["demucs_model"])),
        ("Whisper", lambda: __import__("faster_whisper").WhisperModel(cfg["lyrics"]["whisper_model"], device="cpu", compute_type="int8")),
        ("CLAP", lambda: (__import__("transformers").ClapModel.from_pretrained(cfg["clap"]["clap_model"]),
                          __import__("transformers").ClapProcessor.from_pretrained(cfg["clap"]["clap_model"]))),
        ("Beat This!", lambda: __import__("beat_this.inference", fromlist=["load_checkpoint"]).load_checkpoint("final0")),
    ]
    for name, fn in steps:
        try:
            fn()
            print(f"{name}: weights cached")
        except Exception as e:
            print(f"{name}: skipped ({type(e).__name__}: {e})")


def _check():
    import shutil
    comps = [("ffmpeg (decoding mp3/m4a/aac)", lambda: shutil.which("ffmpeg") or (_ for _ in ()).throw(ImportError())),
             ("essentia (HPCP, key, danceability)", lambda: __import__("essentia")),
             ("essentia-tensorflow (pretrained models)", lambda: __import__("essentia.standard", fromlist=["TensorflowPredict2D"]).TensorflowPredict2D),
             ("torch", lambda: __import__("torch")),
             ("demucs (source separation)", lambda: __import__("demucs")),
             ("opensmile (voice quality)", lambda: __import__("opensmile")),
             ("faster-whisper (lyrics)", lambda: __import__("faster_whisper")),
             ("transformers (CLAP)", lambda: __import__("transformers")),
             ("vaderSentiment (English lyric sentiment)", lambda: __import__("vaderSentiment")),
             ("beat_this (neural beats/downbeats, optional)", lambda: __import__("beat_this")),
             ("allin1 (learned structure labels, optional)", lambda: __import__("allin1"))]
    for name, fn in comps:
        try:
            fn()
            print(f"  [ok]      {name}")
        except Exception:
            print(f"  [missing] {name}")
    try:
        import torch
        print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
