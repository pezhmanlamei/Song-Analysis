"""Run report for the FMA test: module status, feature coverage, and checks of pipeline output
against the labels that come with the Free Music Archive songs.

    python ci/report_fma_sample.py <results folder> <manifest.csv> <report folder>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULES = ["loudness", "timbre", "rhythm", "harmony", "structure", "vocals", "voice_quality", "lyrics",
           "essentia_models", "clap"]
# FMA top-level genre -> Discogs parent genre predicted by the Essentia model
GENRE_TO_DISCOGS = {"Rock": "Rock", "Electronic": "Electronic", "Hip-Hop": "Hip Hop", "Pop": "Pop",
                    "Folk": "Folk, World, & Country", "Country": "Folk, World, & Country", "Jazz": "Jazz",
                    "Blues": "Blues", "Soul-RnB": "Funk / Soul", "Classical": "Classical"}


def main(res: Path, manifest: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(res / "features.csv")
    man = pd.read_csv(manifest)
    lines = ["# FMA 100-song pipeline test", ""]

    # ---- module status
    rows = []
    for tid in man["track_id"]:
        p = res / "cache" / f"{tid}.json"
        st = json.loads(p.read_text())["modules"] if p.exists() else {}
        row = {"track_id": tid}
        for m in MODULES:
            v = st.get(m)
            row[m] = "missing" if v is None else ("ok" if v["status"] == "ok" else "FAILED: " + v.get("error", "")[:150])
        rows.append(row)
    status = pd.DataFrame(rows)
    status.to_csv(out / "module_status.csv", index=False)
    n_ok = {m: int((status[m] == "ok").sum()) for m in MODULES}
    failed = status.melt(id_vars="track_id").query("value != 'ok'")
    lines += [f"Songs in manifest: {len(man)}; rows in features.csv: {len(f)}; columns: {f.shape[1]}", "",
              "| module | songs ok |", "|---|---|"] + [f"| {m} | {n_ok[m]}/{len(man)} |" for m in MODULES] + [""]
    if len(failed):
        lines += [f"**{len(failed)} module failures** (see module_status.csv):", ""]
        lines += [f"- {r.track_id} / {r.variable}: {r.value}" for r in failed.head(30).itertuples()] + [""]
    else:
        lines += ["No module failures.", ""]

    # ---- coverage
    cov = pd.DataFrame({"feature": f.columns, "songs_with_value": f.notna().sum().values})
    cov.to_csv(out / "feature_coverage.csv", index=False)
    empty = cov[cov.songs_with_value == 0].feature.tolist()
    lines += [f"Feature columns empty for every song: {len(empty)}" + (f" ({', '.join(empty[:15])})" if empty else ""), ""]

    # ---- checks against FMA labels
    d = f.merge(man[["track_id", "genre", "fma_language", "fma_instrumental_flag", "album_title", "fma_listens"]],
                on="track_id", how="left", suffixes=("", "_man"))
    d = d[d.track_id.isin(status.track_id[(status[MODULES] == "ok").any(axis=1)])]   # songs actually processed
    lines += [f"Label checks below use the {len(d)} processed songs.", ""]
    checks = []
    flagged = d[d.fma_instrumental_flag == 1]
    if "is_instrumental" in d and flagged.is_instrumental.notna().any():
        fl = flagged[flagged.is_instrumental.notna()]
        checks.append(("FMA-flagged instrumental songs detected as instrumental by separation",
                       f"{int((fl.is_instrumental == 1).sum())}/{len(fl)}"))
        if "voice_prob" in d and flagged.voice_prob.notna().any():
            fl = flagged[flagged.voice_prob.notna()]
            checks.append(("... and by the Essentia voice model (p < 0.5)",
                           f"{int((fl.voice_prob < 0.5).sum())}/{len(fl)}"))
    if "lyrics_language" in d:
        both = d[d.fma_language.notna() & d.lyrics_language.notna() & (d.lyrics_source == "whisper")]
        if len(both):
            checks.append(("Whisper language = FMA language tag (songs with both)",
                           f"{int((both.lyrics_language == both.fma_language).sum())}/{len(both)}"))
    if "genre_parent_top1" in d and d.genre_parent_top1.notna().any():
        dg = d[d.genre_parent_top1.notna()]
        exp = dg.genre.map(GENRE_TO_DISCOGS)
        checks.append(("Essentia top genre matches the FMA genre",
                       f"{int((dg.genre_parent_top1 == exp).sum())}/{len(dg)}"))
        acc = (dg.assign(hit=(dg.genre_parent_top1 == exp)).groupby("genre").hit.mean().round(2).to_dict())
        checks.append(("... per FMA genre", ", ".join(f"{k} {v:.0%}" for k, v in acc.items())))
    if "clap_ref_live_margin" in d:
        dc = d[d.clap_ref_live_margin.notna()]
        live = dc.album_title.fillna("").str.contains(r"\blive\b", case=False)
        if live.any() and (~live).any():
            checks.append(("CLAP live-vs-studio margin, albums titled 'Live' vs others (higher = more live)",
                           f"{dc[live].clap_ref_live_margin.mean():.3f} (n={int(live.sum())}) vs "
                           f"{dc[~live].clap_ref_live_margin.mean():.3f} (n={int((~live).sum())})"))
    if "time_signature_est" in d and "beats_per_bar" in d:
        both = d[d.beats_per_bar.notna() & d.time_signature_est.notna()]
        if len(both):
            checks.append(("Meter: Beat This! beats_per_bar = periodicity estimate",
                           f"{int((both.beats_per_bar == both.time_signature_est).sum())}/{len(both)}"))
    if "vocal_detectors_disagree" in d:
        checks.append(("Songs where separation and the voice model disagree on vocals (listen to these)",
                       f"{int((d.vocal_detectors_disagree == 1).sum())}/{int(d.vocal_detectors_disagree.notna().sum())}"))
    pd.DataFrame(checks, columns=["check", "result"]).to_csv(out / "label_checks.csv", index=False)
    lines += ["## Checks against FMA labels", "", "| check | result |", "|---|---|"] + \
             [f"| {c} | {r} |" for c, r in checks] + ["",
             "FMA's instrumental flag is only reliable when set to 1 (0 means 'not flagged').", ""]

    # ---- per-genre overview of a few features
    keep = [c for c in ["tempo_bpm", "loudness_integrated_lufs", "open_energy", "open_danceability", "open_valence",
                        "open_acousticness", "open_instrumentalness", "open_speechiness", "vocal_presence_ratio",
                        "chord_change_rate_per_min", "n_sections"] if c in d]
    g = d.groupby("genre")[keep].median().round(2).dropna(how="all")
    g.to_csv(out / "genre_medians.csv")
    lines += ["## Median of selected features by genre", "", g.to_markdown() if hasattr(g, "to_markdown") else g.to_string(), ""]
    (out / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
