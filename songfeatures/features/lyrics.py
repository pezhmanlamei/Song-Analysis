"""Lyrics: automatic transcription of the separated vocal (faster-whisper) or provided
lyric files, plus text features (repetitiveness, lexical diversity, pronouns, sentiment).

Caveats: Whisper can hallucinate on singing; use provided lyrics (manifest column
`lyrics_path`) where possible and validate a sample. Transcripts are saved per track for
auditing; lyrics are copyrighted text, so do not redistribute them.
"""
from __future__ import annotations

import os
import re
import zlib

import numpy as np

from ..audio import SR_HI, Track
from ..utils import NAN

NAME = "lyrics"
INFO = {
    "tool": "faster-whisper (Whisper large-v3 by default) on the vocal stem; VADER sentiment (English)",
    "reference": "Radford et al. (2023) Whisper, ICML; Hutto & Gilbert (2014) VADER; "
                 "Parada-Cabaleiro et al. (2024) lyric repetitiveness/compressibility, Scientific Reports",
}
FEATURES = {
    "lyrics_source": ("provided file or whisper transcription", "text"),
    "lyrics_language": ("Detected (or provided) lyric language, ISO 639-1; empty when Whisper kept < 10 words", "text"),
    "lyrics_language_prob": ("Whisper language-identification probability", "0-1"),
    "lyrics_beam_size": ("Whisper beam size used (auto: 5 with >= 12 GB RAM, else 1)", "count"),
    "lyrics_n_words": ("Number of words", "count"),
    "lyrics_n_unique_words": ("Number of distinct words", "count"),
    "lyrics_mattr50": ("Moving-average type-token ratio (window 50 words); lexical diversity", "0-1"),
    "lyrics_compression_ratio": ("zlib-compressed / raw size; lower = more repetitive lyrics", "ratio"),
    "lyrics_line_repetition_share": ("Share of lines that repeat an earlier line exactly", "proportion"),
    "lyrics_mean_word_length": ("Mean characters per word", "chars"),
    "lyrics_words_per_vocal_s": ("Words per second of vocal activity (delivery rate)", "1/s"),
    "lyrics_syllables_per_vocal_s": ("Approx. syllables (vowel groups) per second of vocal activity", "1/s"),
    "lyrics_first_person_rate": ("Share of words that are I/me/my/mine/myself (English only)", "proportion"),
    "lyrics_second_person_rate": ("Share of words that are you/your/yours/yourself (English only)", "proportion"),
    "lyrics_sentiment_compound": ("Mean VADER compound sentiment over lines (English only)", "-1..1"),
    "lyrics_positive_line_share": ("Share of lines with VADER compound >= .05 (English only)", "proportion"),
    "lyrics_negative_line_share": ("Share of lines with VADER compound <= -.05 (English only)", "proportion"),
}
PATTERNS: list = []

_WHISPER = None
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
VOWELS = re.compile(r"[aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüıİœаеёиоуыэюя]+", re.IGNORECASE)


def text_features(lines: list[str], language: str | None) -> dict:
    out = {k: NAN for k in FEATURES if k not in ("lyrics_source", "lyrics_language", "lyrics_language_prob", "lyrics_beam_size",
                                                 "lyrics_words_per_vocal_s", "lyrics_syllables_per_vocal_s")}
    lines = [ln.strip() for ln in lines if ln.strip()]
    text = "\n".join(lines)
    words = [w.lower() for w in WORD_RE.findall(text)]
    out["lyrics_n_words"] = len(words)
    if len(words) < 10:
        return out
    out["lyrics_n_unique_words"] = len(set(words))
    w = 50
    if len(words) >= w:
        out["lyrics_mattr50"] = float(np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)]))
    else:
        out["lyrics_mattr50"] = len(set(words)) / len(words)
    norm = " ".join(words).encode("utf-8")
    out["lyrics_compression_ratio"] = len(zlib.compress(norm, 9)) / len(norm)
    seen, rep = set(), 0
    for ln in lines:
        key = " ".join(WORD_RE.findall(ln.lower()))
        rep += key in seen
        seen.add(key)
    out["lyrics_line_repetition_share"] = rep / len(lines)
    out["lyrics_mean_word_length"] = float(np.mean([len(x) for x in words]))
    if (language or "").startswith("en"):
        fp = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
        sp = {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "you'd", "ya"}
        out["lyrics_first_person_rate"] = sum(x in fp for x in words) / len(words)
        out["lyrics_second_person_rate"] = sum(x in sp for x in words) / len(words)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            comp = np.array([sia.polarity_scores(ln)["compound"] for ln in lines])
            out["lyrics_sentiment_compound"] = float(comp.mean())
            out["lyrics_positive_line_share"] = float(np.mean(comp >= 0.05))
            out["lyrics_negative_line_share"] = float(np.mean(comp <= -0.05))
        except ImportError:
            pass
    return out


def available_memory_gb() -> float:
    """Usable RAM: the smaller of physical memory and the container (cgroup) limit."""
    vals = []
    try:
        vals.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, AttributeError):
        pass
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 2**60:
                vals.append(int(v) / 2**30)
        except OSError:
            pass
    return min(vals) if vals else float("nan")


def beam_size(cfg: dict) -> int:
    """Whisper beam size. 'auto': 5 (Whisper default) with >= 12 GB RAM, else 1, because beam search
    with large-v3 on CPU peaks at ~6 GB (vs ~4 GB greedy)."""
    b = cfg.get("beam_size", "auto")
    if b == "auto":
        mem = available_memory_gb()
        return 5 if (mem != mem or mem >= 12) else 1
    return int(b)


def transcribe(audio16: np.ndarray, cfg: dict):
    global _WHISPER
    from faster_whisper import WhisperModel
    if _WHISPER is None:
        dev = cfg.get("device", "auto")
        if dev == "auto":
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        _WHISPER = WhisperModel(cfg.get("whisper_model", "large-v3"), device=dev,
                                compute_type="float16" if dev == "cuda" else "int8")
    segments, info = _WHISPER.transcribe(audio16, language=cfg.get("language") or None, beam_size=beam_size(cfg),
                                         vad_filter=True, word_timestamps=False,
                                         condition_on_previous_text=False)
    lines = [s.text.strip() for s in segments
             if s.no_speech_prob < cfg.get("max_no_speech_prob", 0.6) and s.avg_logprob > cfg.get("min_avg_logprob", -1.0)]
    return lines, info.language, float(info.language_probability)


def extract(track: Track, cfg: dict) -> dict:
    import librosa

    out = {"lyrics_source": NAN, "lyrics_language": NAN, "lyrics_language_prob": NAN}
    lp = track.meta.get("lyrics_path")
    lang = track.meta.get("lyrics_language") or cfg.get("language")
    if isinstance(lp, str) and lp.strip():
        with open(lp, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        out["lyrics_source"], out["lyrics_language"] = "provided", lang or NAN
    elif cfg.get("transcribe", True):
        from .vocals import vocal_activity
        va = vocal_activity(track, cfg)
        if va["active"].mean() < 0.05:
            out.update(lyrics_source="instrumental", lyrics_n_words=0)
            return out
        a16 = librosa.resample(va["v_hi"].astype(np.float32), orig_sr=SR_HI, target_sr=16000)
        lines, lang, prob = transcribe(a16, cfg)
        out.update(lyrics_source="whisper", lyrics_language=lang, lyrics_language_prob=prob,
                   lyrics_beam_size=beam_size(cfg))
        if track.artifact_dir is not None:
            from pathlib import Path
            Path(track.artifact_dir).mkdir(parents=True, exist_ok=True)
            (Path(track.artifact_dir) / "lyrics_transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    else:
        return out
    out.update(text_features(lines, lang if isinstance(lang, str) else None))
    if out["lyrics_source"] == "whisper" and out.get("lyrics_n_words", 0) < 10:
        # Whisper always guesses a language, even for humming or a choir; without words it means nothing
        out["lyrics_language"], out["lyrics_language_prob"] = NAN, NAN
    # vocal seconds from the shared (cached) vocal-activity mask: the vocals module may have run in another
    # process, so its in-memory value is not available here
    vs = NAN
    if out["lyrics_source"] == "whisper":
        from ..audio import HOP, SR
        vs = float(va["active"].sum() * HOP / SR)
    vs = track.get("vocal_seconds", lambda: vs)
    if np.isfinite(vs) and vs > 0 and out.get("lyrics_n_words", 0):
        text = " ".join(lines)
        out["lyrics_words_per_vocal_s"] = out["lyrics_n_words"] / vs
        out["lyrics_syllables_per_vocal_s"] = len(VOWELS.findall(text)) / vs
    return out
