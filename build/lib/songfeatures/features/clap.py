"""Zero-shot concept scores with CLAP (contrastive language-audio pretraining): measure
theory-driven constructs you describe in words ("nostalgic", "cinematic", ...).

Each concept is a text prompt, optionally with a contrast prompt. With a contrast the
score is P(prompt | prompt vs contrast) from CLAP's calibrated logits (0-1); without, the
raw cosine similarity (z-score these across your corpus). Always validate concept scores
against human ratings on a subsample before using them as measures.
"""
from __future__ import annotations

import numpy as np

from ..audio import SR_HI, Track
from ..utils import slug

NAME = "clap"
INFO = {
    "tool": "LAION CLAP via Hugging Face transformers (default laion/larger_clap_general)",
    "reference": "Wu et al. (2023) Large-scale contrastive language-audio pretraining with feature fusion and "
                 "keyword-to-caption augmentation, ICASSP",
}
# Fixed reference prompts (always computed; used by the open_* Spotify-equivalent layer)
REFERENCE = [
    ("ref_live", "a live concert recording with an audience", "a studio recording", "mean"),
    ("ref_crowd_max", "an audience cheering and applauding", "music without an audience", "max"),
    ("ref_speech", "spoken words or rap vocals", "sung vocals with a melody", "mean"),
    ("ref_happy", "a happy, cheerful song", "a sad, melancholic song", "mean"),
    ("ref_acoustic", "an acoustic song with acoustic instruments", "an electronic song with synthesizers", "mean"),
    ("ref_energetic", "an energetic, intense song", "a calm, gentle song", "mean"),
]
FEATURES: dict = {
    f"clap_{n}": (f"CLAP P('{p}' vs '{c}'), {'max over 10-s chunks' if agg == 'max' else 'song-level'} (fixed reference prompt)", "0-1")
    for n, p, c, agg in REFERENCE
}
PATTERNS = [(r"^clap_.*_margin$", "CLAP cosine(audio, prompt) - cosine(audio, contrast): continuous concept score "
             "(use this for analysis; the 0-1 probability saturates)", "cosine difference"),
            (r"^clap_", "CLAP zero-shot concept score for the prompt defined in config.yaml (clap.concepts)", "0-1 or cosine")]

DEFAULT_CONCEPTS = [
    {"name": "happy", "prompt": "a happy, cheerful song", "contrast": "a sad, melancholic song"},
    {"name": "energetic", "prompt": "an energetic, intense song", "contrast": "a calm, gentle song"},
    {"name": "nostalgic", "prompt": "a nostalgic, sentimental song", "contrast": "a modern, futuristic song"},
    {"name": "cinematic", "prompt": "a cinematic, epic song", "contrast": "an intimate, minimal song"},
    {"name": "romantic", "prompt": "a romantic love song", "contrast": "a song about partying"},
    {"name": "dark", "prompt": "a dark, ominous song", "contrast": "a bright, uplifting song"},
    {"name": "lofi", "prompt": "a lo-fi recording", "contrast": "a polished, high-fidelity studio production"},
    {"name": "live", "prompt": "a live concert recording with audience", "contrast": "a studio recording"},
    {"name": "acoustic", "prompt": "an acoustic song with acoustic instruments", "contrast": "an electronic song with synthesizers"},
    {"name": "catchy", "prompt": "a catchy song with a memorable hook", "contrast": "an experimental song without a clear melody"},
]
_MODEL = None


def _load(cfg):
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import ClapModel, ClapProcessor
        dev = cfg.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = cfg.get("clap_model", "laion/larger_clap_general")
        _MODEL = (ClapModel.from_pretrained(name).to(dev).eval(), ClapProcessor.from_pretrained(name), dev)
    return _MODEL


def _emb(out):
    """transformers <5 returns the projected embedding tensor; >=5 returns an output object whose
    pooler_output is that same projected embedding."""
    import torch
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _proc_audio(proc, chunks):
    try:
        return proc(audio=chunks, sampling_rate=48000, return_tensors="pt")
    except TypeError:
        return proc(audios=chunks, sampling_rate=48000, return_tensors="pt")


def extract(track: Track, cfg: dict) -> dict:
    import librosa
    import torch

    model, proc, dev = _load(cfg)
    y48 = librosa.resample(track.y_hi, orig_sr=SR_HI, target_sr=48000).astype(np.float32)
    win = int(float(cfg.get("chunk_s", 10.0)) * 48000)
    chunks = [y48[i:i + win] for i in range(0, max(len(y48) - win // 2, 1), win)]
    embs = []
    with torch.no_grad():
        for i in range(0, len(chunks), 8):
            inp = {k: v.to(dev) for k, v in _proc_audio(proc, chunks[i:i + 8]).items()}
            embs.append(torch.nn.functional.normalize(_emb(model.get_audio_features(**inp)), dim=-1).cpu())
        A = torch.cat(embs)                                # chunks x d
        a_mean = torch.nn.functional.normalize(A.mean(0, keepdim=True), dim=-1)
        concepts = cfg.get("concepts") or DEFAULT_CONCEPTS
        prompts = [c["prompt"] for c in concepts] + [c["contrast"] for c in concepts if c.get("contrast")]
        prompts += [p for _, p, c, _ in REFERENCE] + [c for _, p, c, _ in REFERENCE]
        prompts = list(dict.fromkeys(prompts))
        tin = {k: v.to(dev) for k, v in proc(text=prompts, return_tensors="pt", padding=True).items()}
        T = torch.nn.functional.normalize(_emb(model.get_text_features(**tin)), dim=-1).cpu()
        # Guard: a broken checkpoint maps every prompt to (almost) the same embedding
        off = (T @ T.T)[~torch.eye(len(prompts), dtype=torch.bool)].mean()
        if float(off) > 0.95:
            raise RuntimeError(f"CLAP text embeddings collapsed (mean prompt similarity {float(off):.3f}); the "
                               f"checkpoint '{cfg.get('clap_model')}' is unusable - use laion/larger_clap_general")
        # Temperature for P(prompt vs contrast). The Hugging Face larger_clap_* checkpoints store
        # logit_scale ~= 0 (exp ~= 1), which would squash every score to ~0.5, so a fixed
        # CLIP/CLAP-convention temperature (default 100) is used unless the model carries a real one.
        learned = float(model.logit_scale_a.exp().cpu()) if hasattr(model, "logit_scale_a") else 0.0
        scale = learned if learned >= 10 else float(cfg.get("temperature", 100.0))
    out = {}
    tmap = dict(zip(prompts, T))
    for c in concepts:
        pos = float(a_mean[0] @ tmap[c["prompt"]])
        if c.get("contrast"):
            neg = float(a_mean[0] @ tmap[c["contrast"]])
            out[f"clap_{slug(c['name'])}"] = float(1 / (1 + np.exp(-scale * (pos - neg))))
            # continuous score for analysis: probabilities saturate near 0/1 at the model temperature
            out[f"clap_{slug(c['name'])}_margin"] = pos - neg
        else:
            out[f"clap_{slug(c['name'])}"] = pos
    for name, p, c, agg in REFERENCE:
        if agg == "max":   # per-chunk probabilities (e.g. applause at the end of a live track)
            d = (A @ tmap[p] - A @ tmap[c]).numpy()
            out[f"clap_{name}"] = float(np.max(1 / (1 + np.exp(-scale * d))))
            out[f"clap_{name}_margin"] = float(np.max(d))
        else:
            d = float(a_mean[0] @ tmap[p] - a_mean[0] @ tmap[c])
            out[f"clap_{name}"] = float(1 / (1 + np.exp(-scale * d)))
            out[f"clap_{name}_margin"] = d
    track.embeddings["clap_mean"] = a_mean[0].numpy()
    return out
