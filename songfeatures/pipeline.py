"""Batch runner: manifest -> per-track cached features -> corpus features -> merged table.

Pass 1  signal-processing modules, parallel over tracks (CPU processes)
Pass 2  model-based modules, one module at a time over all tracks (each model loaded once,\n        then released before the next, so peak memory stays at one model; GPU if present)
Pass 3  corpus-level modules (surprise, novelty) over all tracks
Every module result is cached per track (results/cache/<id>.json), so an interrupted run
resumes where it stopped; a failure in one module never stops the others.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import AudioLoadError, Track
from .utils import safe_id

CORE = ["loudness", "timbre", "rhythm", "harmony", "structure"]
MODEL = ["vocals", "voice_quality", "lyrics", "essentia_models", "clap"]
CORPUS = ["surprise", "novelty"]
VOCAL_DEPENDENT = {"voice_quality", "lyrics"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma", ".alac"}

DEFAULTS = {
    "general": {"min_duration_s": 10.0, "device": "auto", "chroma_method": "auto", "chroma_source": "mix",
                "beat_tracker": "librosa", "isolate_model_steps": True},
    "modules": {**{m: True for m in CORE + MODEL + CORPUS}, "open_features": True},
    "loudness": {"true_peak": True},
    "timbre": {},
    "rhythm": {"meter_margin": 0.1, "meter_backend": None},
    "harmony": {"key_method": "auto", "essentia_key_profile": "bgate", "key_profile": "krumhansl",
                "key_window_beats": 16, "key_hop_beats": 4, "key_self_transition": 0.9, "key_min_windows": 3,
                "chord_self_transition": 0.6, "roughness_frame_step": 4},
    "structure": {"backend": "laplacian", "n_section_types": "auto", "min_section_beats": 8, "drop_db": 6.0},
    "vocals": {"demucs_model": "htdemucs", "demucs_shifts": 1, "keep_stems": "vocals",
               "vocal_rel_db": -15.0, "vocal_abs_db": -50.0, "pitch": True, "speech_sung_threshold": 0.2,
               "voice_gate": "auto", "voice_gate_threshold": 0.5},
    "voice_quality": {"min_vocal_seconds": 5.0},
    "lyrics": {"transcribe": True, "whisper_model": "large-v3", "language": None,
               "max_no_speech_prob": 0.6, "min_avg_logprob": -1.0},
    "essentia_models": {"essentia_model_dir": "models/essentia", "heads": None},
    "clap": {"clap_model": "laion/larger_clap_general", "chunk_s": 10.0, "concepts": None, "temperature": 100.0},
    "surprise": {"chord_order": 3, "melody_order": 3, "alpha": 0.05, "min_corpus_songs": 20},
    "novelty": {"period_window_years": 1, "knn": 10, "min_corpus_songs": 20},
}


# ------------------------------------------------------------------------------ config
def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None) -> dict:
    cfg = DEFAULTS
    if path:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    return cfg


def module_cfg(cfg: dict, name: str) -> dict:
    c = dict(cfg["general"])
    # the vocal modules use the Essentia voice classifier as a gate, so they need its model folder
    c["essentia_model_dir"] = (cfg.get("essentia_models") or {}).get("essentia_model_dir", "models/essentia")
    if name in VOCAL_DEPENDENT:
        c.update(cfg.get("vocals", {}))
    c.update(cfg.get(name) or {})
    return c


def _module(name):
    import importlib
    pkg = "corpus" if name in CORPUS else "features"
    return importlib.import_module(f"songfeatures.{pkg}.{name}")


# ------------------------------------------------------------------------------ manifest
def load_manifest(manifest: str | None = None, audio_dir: str | None = None) -> pd.DataFrame:
    if manifest:
        df = pd.read_csv(manifest)
        if "audio_path" not in df:
            raise ValueError("manifest needs an 'audio_path' column")
        base = Path(manifest).resolve().parent
        df["audio_path"] = [str(p if Path(p).is_absolute() else base / p) for p in df["audio_path"]]
        for col in ("vocals_path", "lyrics_path"):
            if col in df:
                df[col] = [str(p if Path(str(p)).is_absolute() else base / str(p)) if isinstance(p, str) and p.strip()
                           else None for p in df[col]]
    elif audio_dir:
        files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT)
        df = pd.DataFrame({"audio_path": [str(p.resolve()) for p in files]})
    else:
        raise ValueError("give --manifest or --audio-dir")
    if "track_id" not in df:
        df["track_id"] = [Path(p).stem for p in df["audio_path"]]
    df["track_id"] = df["track_id"].astype(str)
    if df["track_id"].duplicated().any():
        dup = df["track_id"][df["track_id"].duplicated()].unique()[:5]
        raise ValueError(f"duplicate track_id values, e.g. {list(dup)}")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------ per track
def _cache_path(out: Path, tid: str) -> Path:
    return out / "cache" / f"{safe_id(tid)}.json"


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"modules": {}}


def _save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, allow_nan=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def process_track(row: dict, modules: list[str], cfg: dict, out: str, force: bool = False) -> dict:
    out = Path(out)
    tid = str(row["track_id"])
    cpath = _cache_path(out, tid)
    state = _load_state(cpath)
    state.update(track_id=tid, audio_path=row["audio_path"])
    todo = [m for m in modules if force or state["modules"].get(m, {}).get("status") != "ok"]
    summary = {"track_id": tid, "ran": [], "errors": []}
    if not todo:
        return summary
    meta = {k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
    track = Track(tid, row["audio_path"], meta=meta, cfg=cfg["general"],
                  artifact_dir=out / "artifacts" / safe_id(tid))
    for m in todo:
        t0 = time.time()
        try:
            feats = _module(m).extract(track, module_cfg(cfg, m))
            state["modules"][m] = {"status": "ok", "seconds": round(time.time() - t0, 2), "features": feats}
            summary["ran"].append(m)
        except AudioLoadError as e:
            for mm in todo:
                state["modules"][mm] = {"status": "error", "error": f"audio: {e}"}
            summary["errors"].append(("audio", str(e)))
            break
        except Exception as e:  # keep going with the other modules
            tb = traceback.format_exc(limit=3)
            state["modules"][m] = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": tb}
            summary["errors"].append((m, f"{type(e).__name__}: {e}"))
        _save_state(cpath, state)
    try:
        track.save_artifacts()
    except Exception as e:
        summary["errors"].append(("artifacts", str(e)))
    _save_state(cpath, state)
    return summary


# ------------------------------------------------------------------------------ run
def _progress(it, total, desc):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc)
    except ImportError:
        return it


def run(manifest_df: pd.DataFrame, out_dir: str, cfg: dict, workers: int = 1, modules: list[str] | None = None,
        force: bool = False, log=print, finalize: bool = True) -> dict:
    """finalize=False (used for shards) runs the per-track passes only; run `merge` afterwards."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enabled = [m for m in CORE + MODEL + CORPUS if cfg["modules"].get(m, False) and (modules is None or m in modules)]
    enabled, skipped = preflight(enabled, cfg, manifest_df, log)
    core = [m for m in CORE if m in enabled]
    model = [m for m in MODEL if m in enabled]
    corpus = [m for m in CORPUS if m in enabled]
    rows = manifest_df.to_dict(orient="records")
    started = datetime.now(timezone.utc).isoformat()
    errors = []

    if core:
        log(f"Pass 1: {', '.join(core)} on {len(rows)} tracks with {workers} worker(s)")
        if workers > 1:
            errors += _core_pool(rows, core, cfg, str(out), force, workers)
        else:
            for r in _progress(rows, len(rows), "core"):
                errors += [(r["track_id"], *e) for e in process_track(r, core, cfg, str(out), force)["errors"]]

    if model:
        isolate = cfg["general"].get("isolate_model_steps", True)
        log(f"Pass 2: {', '.join(model)} (one model at a time over all tracks"
            f"{', each in a fresh process' if isolate else ', then released'})")
        for m in model:
            if isolate:   # all memory of the previous model is returned to the OS; all CPU cores
                errors += _model_step_isolated(rows, m, cfg, str(out), force, log)
            else:
                errors += _model_step(rows, m, cfg, str(out), force)
                release_models(m)

    if not finalize:
        log(f"Shard done ({len(rows)} tracks, {len(errors)} module errors). Run `merge` when all shards finish.")
        return {"n_tracks": len(rows), "n_errors": len(errors)}

    table = collect(manifest_df, out)
    if cfg["modules"].get("open_features", True):
        from .derived import add_spotify_equivalents
        table = add_spotify_equivalents(table)
    if corpus:
        log(f"Pass 3: {', '.join(corpus)} over the corpus")
        corp = corpus_stage(manifest_df, table, out, cfg, corpus)
        if not corp.empty:
            table = table.merge(corp, on="track_id", how="left")

    table.to_csv(out / "features.csv", index=False)
    from .codebook import build_codebook
    build_codebook(list(table.columns)).to_csv(out / "codebook.csv", index=False)
    err_df = collect_errors(out, manifest_df)
    err_df.to_csv(out / "errors.csv", index=False)
    info = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "n_tracks": len(rows), "n_module_errors": int(len(err_df)), "modules": enabled,
            "skipped_modules": skipped,
            "config": cfg, "versions": _versions(), "python": platform.python_version(),
            "platform": platform.platform()}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    log(f"Done: {out / 'features.csv'} ({table.shape[0]} rows x {table.shape[1]} columns), "
        f"{len(err_df)} module errors -> errors.csv")
    return info


def _set_threads(n: int):
    """Process initializer: thread count for BLAS/OpenMP/numba/torch in this process."""
    import faulthandler
    faulthandler.enable()   # a native crash (segfault) prints the Python stack to stderr
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(n)


CRASH_MSG = ("WorkerCrashed: the worker process died on this song (out of memory or a fault in a native "
             "library); the other songs were not affected")


def _mark_crashed(out: str, row: dict, modules: list[str]):
    """Record a crash in the song's cache so it shows up in errors.csv / module_status."""
    cpath = _cache_path(Path(out), str(row["track_id"]))
    state = _load_state(cpath)
    state.update(track_id=str(row["track_id"]), audio_path=row["audio_path"])
    for m in modules:
        if state["modules"].get(m, {}).get("status") != "ok":
            state["modules"][m] = {"status": "error", "error": CRASH_MSG}
    _save_state(cpath, state)


def _pool(n_workers: int, n_threads: int):
    import multiprocessing as mp
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                               initializer=_set_threads, initargs=(n_threads,))


def _core_pool(rows, core, cfg, out, force, workers) -> list:
    """Pass 1 in a process pool (1 thread per worker). If a worker dies, the songs that had not finished
    are retried one at a time, each in its own process, so one bad song cannot take down the others."""
    from concurrent.futures.process import BrokenProcessPool
    errors, done = [], set()
    try:
        with _pool(workers, 1) as ex:
            futs = {ex.submit(process_track, r, core, cfg, out, force): r for r in rows}
            for f in _progress(as_completed(futs), len(futs), "core"):
                res = f.result()
                done.add(res["track_id"])
                errors += [(res["track_id"], *e) for e in res["errors"]]
    except BrokenProcessPool:
        left = [r for r in rows if str(r["track_id"]) not in done]
        print(f"A worker process died; retrying {len(left)} song(s) one at a time", flush=True)
        for r in left:
            try:
                with _pool(1, os.cpu_count() or 1) as ex:
                    res = ex.submit(process_track, r, core, cfg, out, force).result()
                errors += [(res["track_id"], *e) for e in res["errors"]]
            except BrokenProcessPool:
                _mark_crashed(out, r, core)
                errors.append((str(r["track_id"]), "core", CRASH_MSG))
    return errors


def _model_step_isolated(rows, m, cfg, out, force, log) -> list:
    """One model module over all songs in a fresh process (all CPU threads). Songs run in order and each
    result is cached at once, so after a crash the song being processed is the first unfinished one:
    it is marked as crashed and the step restarts in a new process with the remaining songs."""
    from concurrent.futures.process import BrokenProcessPool
    errors, todo = [], list(rows)
    while todo:
        try:
            with _pool(1, os.cpu_count() or 1) as ex:
                errors += ex.submit(_model_step, todo, m, cfg, out, force).result()
            break
        except BrokenProcessPool:
            pending = [r for r in todo if _load_state(_cache_path(Path(out), str(r["track_id"])))
                       ["modules"].get(m, {}).get("status") not in ("ok", "error")]
            if not pending:
                break
            bad = pending[0]
            log(f"{m}: worker died on {bad['track_id']}; marking it and continuing with {len(pending) - 1} song(s)")
            _mark_crashed(out, bad, [m])
            errors.append((str(bad["track_id"]), m, CRASH_MSG))
            todo = pending[1:]
    return errors


def _model_step(rows: list[dict], m: str, cfg: dict, out: str, force: bool) -> list:
    """Run one model module over all tracks (in the current process)."""
    import gc
    errors = []
    for r in _progress(rows, len(rows), m):
        errors += [(r["track_id"], *e) for e in process_track(r, [m], cfg, out, force)["errors"]]
        gc.collect()
        _trim_memory()   # give freed per-song buffers back to the OS (keeps the model loaded)
    return errors


def _trim_memory():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def release_models(name: str):
    """Free a module's cached models (and the shared Essentia graphs) before the next module loads."""
    import gc
    for mod_name in (name, "essentia_models"):
        try:
            mod = _module(mod_name)
        except Exception:
            continue
        for attr, empty in (("_MODELS", {}), ("_ALGOS", {}), ("_WHISPER", None), ("_MODEL", None), ("_SMILE", None)):
            if hasattr(mod, attr):
                setattr(mod, attr, type(empty)() if isinstance(empty, dict) else None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def preflight(enabled: list[str], cfg: dict, manifest_df: pd.DataFrame, log=print):
    """Disable modules whose dependencies are missing (once, with an install hint) instead of
    failing on every track."""
    import importlib.util

    def can(pkg):
        try:
            return importlib.util.find_spec(pkg) is not None
        except (ImportError, ValueError):
            return False

    def has_col(c):
        return c in manifest_df and manifest_df[c].notna().any()

    sep_ok = (can("torch") and can("demucs")) or has_col("vocals_path")
    hint = "pip install -r requirements-models.txt"
    skip = {}
    if "vocals" in enabled and not sep_ok:
        skip["vocals"] = f"needs torch + demucs for source separation ({hint})"
    if "voice_quality" in enabled:
        if not can("opensmile"):
            skip["voice_quality"] = "needs opensmile (pip install opensmile)"
        elif not sep_ok:
            skip["voice_quality"] = "needs vocal stems (torch + demucs or a vocals_path column)"
    if "lyrics" in enabled and not has_col("lyrics_path"):
        if not cfg["lyrics"].get("transcribe", True):
            skip["lyrics"] = "transcription disabled and no lyrics_path column"
        elif not can("faster_whisper"):
            skip["lyrics"] = f"needs faster-whisper ({hint}) or a lyrics_path column"
        elif not sep_ok:
            skip["lyrics"] = "needs vocal stems (torch + demucs or a vocals_path column)"
    if "essentia_models" in enabled:
        ok = False
        try:
            import essentia.standard as es
            ok = hasattr(es, "TensorflowPredict2D")
        except ImportError:
            pass
        mdir = Path(cfg["essentia_models"].get("essentia_model_dir", "models/essentia")).expanduser()
        if not ok:
            skip["essentia_models"] = "needs essentia-tensorflow (Linux/macOS: pip install essentia-tensorflow)"
        elif not any(mdir.glob("*.pb")):
            skip["essentia_models"] = f"no model files in {mdir} (run: python -m songfeatures download-models)"
    if "clap" in enabled and not (can("torch") and can("transformers")):
        skip["clap"] = f"needs torch + transformers ({hint})"
    if cfg["general"].get("beat_tracker") == "beat_this" and not can("beat_this"):
        log("  ! beat_tracker=beat_this but beat_this is not installed; falling back to librosa")
    for m, why in skip.items():
        log(f"  ! skipping {m}: {why}")
    return [m for m in enabled if m not in skip], skip


def collect(manifest_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for r in manifest_df.to_dict(orient="records"):
        state = _load_state(_cache_path(out, str(r["track_id"])))
        row = dict(r)
        for m in CORE + MODEL:
            mod = state["modules"].get(m)
            if mod and mod.get("status") == "ok":
                row.update(mod["features"])
        rows.append(row)
    return pd.DataFrame(rows)


def collect_errors(out: Path, manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid in manifest_df["track_id"]:
        for m, mod in _load_state(_cache_path(out, str(tid)))["modules"].items():
            if mod.get("status") == "error":
                rows.append({"track_id": tid, "module": m, "error": mod.get("error"), "traceback": mod.get("traceback", "")})
    return pd.DataFrame(rows, columns=["track_id", "module", "error", "traceback"])


def corpus_stage(manifest_df, table, out: Path, cfg, corpus_modules) -> pd.DataFrame:
    from .corpus.novelty import corpus_novelty
    from .corpus.surprise import corpus_surprise
    seqs, embs = {}, {}
    for tid in manifest_df["track_id"]:
        d = out / "artifacts" / safe_id(str(tid))
        if (d / "sequences.json").exists():
            seqs[tid] = json.loads((d / "sequences.json").read_text())
        if (d / "embeddings.npz").exists():
            embs[tid] = dict(np.load(d / "embeddings.npz"))
    res: dict[str, dict] = {}
    if "surprise" in corpus_modules and seqs:
        for t, r in corpus_surprise(seqs, cfg["surprise"]).items():
            res.setdefault(t, {}).update(r)
    if "novelty" in corpus_modules and embs:
        meta = table.set_index("track_id")
        for t, r in corpus_novelty(embs, meta, cfg["novelty"]).items():
            res.setdefault(t, {}).update(r)
    res = {t: r for t, r in res.items() if r}
    if not res:
        log_msg = "corpus stage: not enough songs with sequences/embeddings (min_corpus_songs) - skipped"
        print("  !", log_msg)
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(res, orient="index")
    df.index.name = "track_id"
    df = df.reset_index()
    df.to_csv(out / "corpus_features.csv", index=False)
    return df


def _versions() -> dict:
    v = {}
    for pkg in ("numpy", "scipy", "librosa", "pyloudnorm", "sklearn", "essentia", "opensmile", "torch",
                "demucs", "faster_whisper", "transformers", "beat_this"):
        try:
            mod = __import__(pkg)
            v[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            v[pkg] = None
    return v
