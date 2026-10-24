"""Feature codebook: one row per output column with definition, unit, module, tool and
reference — ready for a methods-section appendix."""
from __future__ import annotations

import re

import pandas as pd

from . import derived
from .corpus import novelty, surprise
from .features import (clap, essentia_models, harmony, loudness, lyrics, rhythm, structure, timbre,
                       vocals, voice_quality)

MODULES = [derived, loudness, timbre, rhythm, harmony, structure, vocals, voice_quality, lyrics,
           essentia_models, clap, surprise, novelty]

FULL_SONG_ONLY = {"structure"}   # features that are not meaningful on 30-s previews


def _lookup(col: str):
    for mod in MODULES:
        if col in mod.FEATURES:
            d, u = mod.FEATURES[col]
            return mod, d, u
    for mod in MODULES:
        for pat, desc, unit in mod.PATTERNS:
            m = re.match(pat, col)
            if m:
                groups = m.groups() or (col.split("_")[-1],)
                return mod, desc.format(*groups), unit
    return None, "", ""


def build_codebook(columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:   # full static codebook: named features + one row per feature family
        rows = []
        for mod in MODULES:
            for c, (d, u) in mod.FEATURES.items():
                rows.append({"feature": c, "module": mod.NAME, "description": d, "unit": u,
                             "full_song_only": int(mod.NAME in FULL_SONG_ONLY),
                             "tool": mod.INFO["tool"], "reference": mod.INFO["reference"]})
            for pat, d, u in mod.PATTERNS:
                label = pat.strip("^$").replace("\\", "") + ("*" if not pat.endswith("$") else "")
                rows.append({"feature": label, "module": mod.NAME, "description": d.replace("{0}", "<x>"),
                             "unit": u, "full_song_only": int(mod.NAME in FULL_SONG_ONLY),
                             "tool": mod.INFO["tool"], "reference": mod.INFO["reference"]})
        return pd.DataFrame(rows)
    rows = []
    for c in columns:
        mod, desc, unit = _lookup(c)
        if mod is None:
            continue
        rows.append({"feature": c, "module": mod.NAME, "description": desc, "unit": unit,
                     "full_song_only": int(mod.NAME in FULL_SONG_ONLY),
                     "tool": mod.INFO["tool"], "reference": mod.INFO["reference"]})
    return pd.DataFrame(rows)
