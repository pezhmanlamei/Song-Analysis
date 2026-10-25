"""Compare Whisper transcription set-ups on selected songs (diagnostic, not part of the pipeline).

    python ci/lyrics_probe.py --manifest ci/fma_sample/manifest.csv --config ci/config_ci.yaml --out probe ID [ID ...]
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from songfeatures.pipeline import load_config, module_cfg  # noqa: E402
from songfeatures.audio import Track, HOP, SR, SR_HI  # noqa: E402
from songfeatures.features import lyrics as L  # noqa: E402
from songfeatures.features.vocals import vocal_activity  # noqa: E402


def runs(active, frame_s, pad=0.25, merge=1.0):
    out, start = [], None
    for i, a in enumerate(np.r_[active.astype(bool), False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            out.append([max(0.0, start * frame_s - pad), i * frame_s + pad]); start = None
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < merge:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--config"); ap.add_argument("--out"); ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(a.manifest).set_index("track_id")
    import librosa
    from faster_whisper import WhisperModel
    wm = WhisperModel(cfg["lyrics"].get("whisper_model", "large-v3"), device="cpu", compute_type="int8")
    rows = []
    for tid in a.ids:
        r = man.loc[tid].to_dict(); r["track_id"] = tid
        tr = Track(tid, r["audio_path"], meta={}, cfg=cfg["general"], artifact_dir=out / "res" / "artifacts" / tid)
        va = vocal_activity(tr, module_cfg(cfg, "vocals"))
        a16 = librosa.resample(va["v_hi"].astype(np.float32), orig_sr=SR_HI, target_sr=16000)
        fs = HOP / SR
        clips = runs(va["active"], fs)
        variants = {
            "A_vad_full": (a16, dict(vad_filter=True)),
            "B_concat": (L.vocal_only_audio(a16, va["active"], fs), dict(vad_filter=False)),
            "C_clip": (a16, dict(vad_filter=False, clip_timestamps=[x for ab in clips for x in ab] or "0")),
            "D_full_novad": (a16, dict(vad_filter=False)),
        }
        for name, (audio, kw) in variants.items():
            t0 = time.time()
            segs, info = wm.transcribe(audio, beam_size=L.beam_size(cfg["lyrics"]), word_timestamps=False,
                                       condition_on_previous_text=False, **kw)
            segs = list(segs)
            f_nsp = [s for s in segs if s.no_speech_prob >= 0.6]
            f_lp = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob <= -1.0]
            f_hal = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0 and L.HALLUCINATION_RE.search(s.text.strip())]
            kept = [s for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0 and not L.HALLUCINATION_RE.search(s.text.strip())]
            words = sum(len(L.WORD_RE.findall(s.text)) for s in kept)
            covered = sum(min(s.end, len(audio) / 16000) - s.start for s in kept)
            rows.append(dict(track_id=tid, artist=r.get("artist"), variant=name, words=words, segments=len(segs), kept=len(kept),
                             drop_no_speech=len(f_nsp), drop_logprob=len(f_lp), drop_hallucination=len(f_hal),
                             lang=info.language, lang_prob=round(info.language_probability, 2),
                             vocal_s=round(float(va["active"].sum() * fs), 1), kept_s=round(covered, 1), sec=round(time.time() - t0)))
            (out / f"{tid}_{name}.txt").write_text("\n".join(f"[{s.start:6.1f}-{s.end:6.1f}] nsp={s.no_speech_prob:.2f} lp={s.avg_logprob:.2f} "
                                                              f"{'KEEP' if s in kept else 'drop'} | {s.text.strip()}" for s in segs))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "probe.csv", index=False)


if __name__ == "__main__":
    main()
