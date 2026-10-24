"""Validation against synthetic songs with known ground truth.

    python tests/run_tests.py            # generates test audio, runs checks, prints PASS/FAIL

Checks the signal-processing modules (and vocal features via a provided stem) plus the
corpus-level surprise/novelty models. Model downloads are not required.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from make_test_songs import main as make_songs  # noqa: E402
from songfeatures.audio import Track  # noqa: E402
from songfeatures.corpus.novelty import corpus_novelty  # noqa: E402
from songfeatures.corpus.surprise import corpus_surprise  # noqa: E402
from songfeatures.features import harmony, loudness, rhythm, structure, timbre, vocals  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def features(path, meta=None):
    tr = Track(Path(path).stem, path, meta=meta or {})
    out = {}
    for m in (loudness, timbre, rhythm, harmony, structure):
        out.update(m.extract(tr, {}))
    return tr, out


def main():
    d = Path(tempfile.mkdtemp(prefix="songfeatures_test_"))
    make_songs(str(d))

    print("song_a: 120 BPM, C major, intro/verse/chorus x2/fade-out, vocal with 5.5 Hz vibrato")
    tr, a = features(d / "song_a.wav")
    check("tempo ~120 BPM", abs(a["tempo_bpm"] - 120) < 2, f"{a['tempo_bpm']:.1f}")
    check("key C major", a["key_label"] == "C major", a["key_label"])
    check("no modulation", a["n_key_changes"] == 0 and a["late_upward_modulation"] == 0)
    check("4 distinct chords", a["n_unique_chords"] == 4, str(a["n_unique_chords"]))
    seq = set(tr.sequences["chords_absolute"]) - {"N"}
    check("chords are C, Am, F, G", seq == {"C", "Am", "F", "G"}, str(sorted(seq)))
    check("fade-out detected (~8 s)", a["ending_type"] == "fade_out" and abs(a["fade_out_duration_s"] - 8) < 2, f"{a['ending_type']} {a['fade_out_duration_s']:.1f} s")
    check("first chorus 20-26 s (true 24 s)", 20 <= a["time_to_first_chorus_s"] <= 26, f"{a['time_to_first_chorus_s']:.1f}")
    check("chorus louder than rest", a["chorus_loudness_contrast_db"] > 1, f"{a['chorus_loudness_contrast_db']:.1f} LU")
    check("stereo width > 0", a["stereo_width"] > 0.01, f"{a['stereo_width']:.3f}")
    check("no leading silence", a["leading_silence_s"] < 0.5, f"{a['leading_silence_s']:.2f}")

    v = vocals.extract(tr2 := Track("a", d / "song_a.wav", meta={"vocals_path": str(d / "song_a_vocals.wav")}), {"voice_gate": "off"})  # synthetic sine "voice" is not a human voice
    check("first vocal at ~8 s", abs(v["time_to_first_vocal_s"] - 8) < 0.5, f"{v['time_to_first_vocal_s']:.2f}")
    check("vocal presence ~0.79", abs(v["vocal_presence_ratio"] - 0.79) < 0.05, f"{v['vocal_presence_ratio']:.2f}")
    check("vibrato rate ~5.5 Hz", abs(v["vibrato_rate_hz"] - 5.5) < 0.5, f"{v['vibrato_rate_hz']:.2f}")
    check("vibrato on ~1/3 of notes", abs(v["vibrato_note_share"] - 1 / 3) < 0.1, f"{v['vibrato_note_share']:.2f}")
    check("melody notes recovered", tr2.sequences["melody_notes_midi"][:6] == [72, 76, 79, 69, 72, 76],
          str(tr2.sequences["melody_notes_midi"][:6]))

    print("song_b: 90 BPM, A minor, off-beat skank, up a whole tone in the last third, cold ending")
    trb, b = features(d / "song_b.wav")
    check("tempo ~90 BPM", abs(b["tempo_bpm"] - 90) < 2, f"{b['tempo_bpm']:.1f}")
    check("key A minor", b["key_label"] == "A minor", b["key_label"])
    check("one key change", b["n_key_changes"] == 1, str(b["n_key_changes"]))
    check("late upward modulation", b["late_upward_modulation"] == 1)
    seqb = set(trb.sequences["chords_absolute"]) - {"N"}
    check("chords Am Dm Em Bm F#m", seqb == {"Am", "Dm", "Em", "Bm", "F#m"}, str(sorted(seqb)))
    check("cold ending, no fade-out", b["ending_type"] == "cold_cut" and b["ends_with_fade"] == 0, b["ending_type"])
    check("more chord vocabulary than song_a", b["chord_entropy_bits"] > a["chord_entropy_bits"])

    print("song_c: mono file")
    _, c = features(d / "song_c_mono.wav")
    check("mono flagged, width 0", c["is_mono_file"] == 1 and c["stereo_width"] == 0)

    print("meter (time signature)")
    check("song_a is 4/4", a["time_signature_est"] == 4, f"margin {a['meter_triple_margin']:.2f}")
    check("song_b is 4/4", b["time_signature_est"] == 4, f"margin {b['meter_triple_margin']:.2f}")
    _, w = features(d / "song_waltz.wav")
    check("synthetic waltz is 3/4", w["time_signature_est"] == 3, f"margin {w['meter_triple_margin']:.2f}")

    print("speech vs singing")
    check("sung vocal is not speech-like", v["speech_like_vocal_share"] < 0.1, f"{v['speech_like_vocal_share']:.2f}")
    check("duration recorded", abs(a["duration_s"] - 81.0) < 1.0, f"{a['duration_s']:.1f} s")

    print("open_* Spotify-equivalent layer")
    from songfeatures.derived import spotify_equivalents
    row = {**a, **v}
    o = spotify_equivalents(row)
    check("open_key/mode = C major (0, 1)", o["open_key"] == 0 and o["open_mode"] == 1)
    check("open_time_signature = 4", o["open_time_signature"] == 4)
    check("open_* in 0-1", all(0 <= o[k] <= 1 for k in ("open_danceability", "open_energy", "open_instrumentalness",
                                                          "open_speechiness")), str({k: round(o[k], 2) for k in o if isinstance(o[k], float)}))
    check("sung song: low speechiness, low instrumentalness", o["open_speechiness"] < 0.1 and o["open_instrumentalness"] < 0.1)
    check("song_b (no vocals stem) instrumentalness unavailable, not guessed",
          np.isnan(spotify_equivalents(b)["open_instrumentalness"]))
    check("louder, busier song_a has more energy than song_b",
          spotify_equivalents(a)["open_energy"] > spotify_equivalents(b)["open_energy"])
    check("sources recorded", "danceability=dfa_div3" in o["open_sources"], o["open_sources"])

    real_audio_checks()

    print("corpus models")
    rng = np.random.default_rng(0)
    seqs, emb, meta = {}, {}, []
    for i in range(40):
        odd = i >= 36
        ch = (list(rng.choice([f"{x}{q}" for x in range(12) for q in "Mm"], 24)) if odd
              else (["0M", "7M", "9m", "5M"] if i % 2 else ["0M", "5M", "7M", "0M"]) * 6)
        mel = rng.choice([-9, -7, 7, 9, 12, -12] if odd else [-2, -1, 0, 1, 2], 60).tolist()
        e = rng.normal(0, .3, 64)
        e[1 if odd else 0] += 3
        seqs[f"s{i}"] = {"chords_key_relative": ch, "melody_intervals": mel}
        emb[f"s{i}"] = {"effnet_mean": e}
        meta.append({"track_id": f"s{i}", "year": 2015 + i % 5, "artist": f"a{i % 6}", "genre": "pop"})
    S = pd.DataFrame(corpus_surprise(seqs, {"min_corpus_songs": 20})).T.astype(float)
    N = pd.DataFrame(corpus_novelty(emb, pd.DataFrame(meta).set_index("track_id"), {"min_corpus_songs": 20})).T.astype(float)
    g = np.r_[np.zeros(36), np.ones(4)]
    check("unusual harmony -> higher chord surprise", S.groupby(g)["chord_ic_mean"].mean().diff().iloc[-1] > 2)
    check("leapy melodies -> higher melodic surprise", S.groupby(g)["melody_ic_mean"].mean().diff().iloc[-1] > 0.5)
    check("atypical sound -> higher novelty", N.groupby(g)["novelty_period_effnet"].mean().diff().iloc[-1] > 0.2)

    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


def real_audio_checks():
    """Optional checks on real recordings from the librosa example collection (needs internet)."""
    try:
        import librosa
        waltz, vibe, speech = librosa.ex("sweetwaltz"), librosa.ex("vibeace"), librosa.ex("libri1")
    except Exception as e:
        print(f"real-audio checks skipped (librosa examples unavailable: {type(e).__name__})")
        return
    print("real recordings (librosa examples)")
    _, w = features(waltz)
    check("real waltz (Sweet Waltz) is 3/4", w["time_signature_est"] == 3, f"margin {w['meter_triple_margin']:.2f}")
    _, e = features(vibe)
    check("real electronic track (Vibe Ace) is 4/4", e["time_signature_est"] == 4, f"margin {e['meter_triple_margin']:.2f}")
    sp = vocals.extract(Track("speech", speech, meta={"vocals_path": speech}, cfg={"min_duration_s": 5}), {})
    check("real speech (LibriSpeech) is speech-like", sp["speech_like_vocal_share"] > 0.7, f"{sp['speech_like_vocal_share']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
