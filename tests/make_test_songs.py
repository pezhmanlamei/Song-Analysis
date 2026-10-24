"""Synthesise test songs with known ground truth for validating the pipeline.

song_a: 120 BPM, C major, intro(8s) V C V C outro-fade; louder chorus with 16th hats; sung-like
        melody with 5.5 Hz vibrato (also written as a separate 'vocals' stem).
song_b: 90 BPM, A minor, off-beat 'skank' chords, no vocals, modulates up a whole tone to
        B minor for the final third, cold (non-faded) ending.
song_c: mono 30 s excerpt of song_a.  song_short: 5 s.  song_broken: not audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
rng = np.random.default_rng(7)


def midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12)


def tone(f, dur, n_harm=6, amp=1.0, vib_rate=0.0, vib_cents=0.0, attack=0.01, release=0.05):
    t = np.arange(int(dur * SR)) / SR
    cents = vib_cents * np.sin(2 * np.pi * vib_rate * t) if vib_rate else np.zeros_like(t)
    inst_f = f * 2 ** (cents / 1200)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    x = sum((1 / h) * np.sin(h * phase) for h in range(1, n_harm + 1))
    env = np.minimum(1, t / attack) * np.minimum(1, (dur - t) / release)
    return amp * x * np.clip(env, 0, 1)


def kick():
    t = np.arange(int(0.2 * SR)) / SR
    f = 50 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18)


def snare():
    t = np.arange(int(0.15 * SR)) / SR
    return (rng.normal(size=t.size) * 0.6 + np.sin(2 * np.pi * 190 * t) * 0.4) * np.exp(-t * 30)


def hat():
    t = np.arange(int(0.04 * SR)) / SR
    n = rng.normal(size=t.size)
    n = np.diff(np.concatenate([[0], n]))  # crude high-pass
    return n * np.exp(-t * 90) * 0.5


def place(buf, x, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= buf.shape[-1]:
        return
    n = min(len(x), buf.shape[-1] - i)
    buf[..., i:i + n] += gain * x[:n]


CHORDS = {"C": [60, 64, 67], "Am": [57, 60, 64], "F": [53, 57, 60], "G": [55, 59, 62],
          "Dm": [50, 53, 57], "Em": [52, 55, 59]}


def transpose(ch, k):
    return [n + k for n in ch]


def song_a(path: Path, vocals_path: Path):
    bpm, beat = 120, 0.5
    bar = 4 * beat
    plan = [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("outro", 4)]
    total = sum(b for _, b in plan) * bar
    mix = np.zeros((2, int(total * SR) + SR))
    voc = np.zeros(mix.shape[1])
    prog = {"intro": ["C", "Am", "F", "G"], "verse": ["C", "Am", "F", "G"],
            "chorus": ["F", "G", "C", "Am"], "outro": ["F", "G", "C", "Am"]}
    t0 = 0.0
    for sec, nbars in plan:
        level = {"intro": 0.12, "verse": 0.25, "chorus": 0.6, "outro": 0.6}[sec]
        for b in range(nbars):
            name = prog[sec][b % 4]
            tb = t0 + b * bar
            for n in CHORDS[name]:
                place(mix[0], tone(midi_hz(n), bar, amp=level * 0.35), tb)
                place(mix[1], tone(midi_hz(n) * 1.002, bar, amp=level * 0.35), tb)
            place(mix, tone(midi_hz(CHORDS[name][0] - 24), bar, n_harm=3, amp=level * 0.6), tb)
            if sec != "intro":
                for k in range(4):
                    tk = tb + k * beat
                    place(mix, kick() if k in (0, 2) else snare(), tk, level * 1.2)
                    div = 4 if sec == "chorus" or sec == "outro" else 2
                    for s in range(div):
                        place(mix, hat(), tk + s * beat / div, level * 0.6)
                # melody: chord tones an octave up, one note per beat, long note on beat 3
                if sec in ("verse", "chorus"):
                    notes = [CHORDS[name][i % 3] + (12 if sec == "verse" else 17) for i in range(3)]
                    durs = [beat, beat, 2 * beat]
                    tt = tb
                    for n, d in zip(notes, durs):
                        v = tone(midi_hz(n), d * 0.95, n_harm=5, amp=0.25 if sec == "verse" else 0.4,
                                 vib_rate=5.5 if d > beat else 0, vib_cents=40 if d > beat else 0,
                                 attack=0.03, release=0.08)
                        place(voc, v, tt)
                        tt += d
        t0 += nbars * bar
    mix[:, : voc.size] += voc
    # fade-out over the outro
    fade_start, fade_end = total - 4 * bar, total
    t = np.arange(mix.shape[1]) / SR
    g = np.clip((fade_end - t) / (fade_end - fade_start), 0, 1)
    mix *= g
    voc *= g
    peak = np.max(np.abs(mix))
    mix, voc = mix / peak * 0.89, voc / peak * 0.89
    sf.write(path, mix.T, SR, subtype="PCM_16")
    sf.write(vocals_path, np.stack([voc, voc]).T, SR, subtype="PCM_16")
    return mix


def song_b(path: Path):
    beat = 60 / 90
    bar = 4 * beat
    nbars = 30
    mix = np.zeros((2, int(nbars * bar * SR) + SR))
    prog = ["Am", "Dm", "Em", "Am"]
    for b in range(nbars):
        k = 2 if b >= 20 else 0                       # modulate up a whole tone for the last third
        name = prog[b % 4]
        tb = b * bar
        ch = transpose(CHORDS[name], k)
        place(mix, tone(midi_hz(ch[0] - 24), bar * 0.95, n_harm=3, amp=0.35), tb)
        for i in range(4):
            off = tb + i * beat + beat / 2           # skank on every off-beat 8th
            for n in ch:
                place(mix[0], tone(midi_hz(n + 12), beat * 0.3, amp=0.18, release=0.03), off)
                place(mix[1], tone(midi_hz(n + 12) * 1.003, beat * 0.3, amp=0.18, release=0.03), off)
            place(mix, hat(), off, 0.4)
        place(mix, kick(), tb + 2 * beat, 0.9)        # one-drop
        place(mix, snare(), tb + 2 * beat, 0.5)
    mix = mix[:, : int(nbars * bar * SR)]
    mix = mix / np.max(np.abs(mix)) * 0.89
    sf.write(path, mix.T, SR, subtype="PCM_16")


def song_waltz(path: Path):
    """3/4 at 150 BPM: bass + kick on beat 1, lighter chord stabs on 2 and 3, chord change every bar."""
    beat = 60 / 150
    bar = 3 * beat
    prog = ["C", "F", "G", "C", "Am", "Dm", "G", "C"]
    nbars = 40
    mix = np.zeros((2, int(nbars * bar * SR) + SR))
    for b in range(nbars):
        tb = b * bar
        ch = CHORDS[prog[b % len(prog)]]
        place(mix, tone(midi_hz(ch[0] - 24), beat * 0.9, n_harm=3, amp=0.6), tb)
        place(mix, kick(), tb, 0.8)
        for k in (1, 2):
            for n in ch:
                place(mix[0], tone(midi_hz(n), beat * 0.5, amp=0.12, release=0.05), tb + k * beat)
                place(mix[1], tone(midi_hz(n) * 1.002, beat * 0.5, amp=0.12, release=0.05), tb + k * beat)
            place(mix, hat(), tb + k * beat, 0.3)
    mix = mix[:, : int(nbars * bar * SR)]
    mix = mix / np.max(np.abs(mix)) * 0.89
    sf.write(path, mix.T, SR, subtype="PCM_16")


def main(out_dir="test_audio"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = song_a(out / "song_a.wav", out / "song_a_vocals.wav")
    song_b(out / "song_b.wav")
    song_waltz(out / "song_waltz.wav")
    sf.write(out / "song_c_mono.wav", a.mean(axis=0)[: 30 * SR], SR, subtype="PCM_16")
    sf.write(out / "song_short.wav", a.mean(axis=0)[: 5 * SR], SR, subtype="PCM_16")
    (out / "song_broken.mp3").write_text("this is not audio")
    print("written to", out.resolve())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_audio")
