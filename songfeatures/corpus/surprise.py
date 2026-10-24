"""Expectation / surprise of harmony and melody, estimated against the analysed corpus.

An interpolated n-gram model (orders 0..N-1, add-alpha smoothing) is trained on all
songs' symbol sequences; each song is scored leave-one-out (its own counts removed), so a
song is "surprising" relative to the rest of the corpus. Per event we compute information
content IC = -log2 P(event | context) (surprise) and entropy H of the predictive
distribution (uncertainty), following the IDyOM logic (Pearce, 2005/2018) in a simplified,
dependency-free form. For publication-grade melodic expectation, IDyOM itself can be run
on the exported note sequences.

Chords are key-relative (scale degree + quality) so the model is transposition invariant.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

NAME = "surprise"
INFO = {
    "tool": "own interpolated n-gram model (leave-one-out over the corpus)",
    "reference": "Pearce (2005; 2018) IDyOM; Cheung et al. (2019) Current Biology - uncertainty x surprise "
                 "in chord sequences predicts pleasure; Gold et al. (2019) J. Neurosci.",
}
FEATURES = {}
PATTERNS = [
    (r"^(chord|melody)_ic_mean$", "Mean information content (surprise) of {0} events given the corpus model", "bits"),
    (r"^(chord|melody)_ic_sd$", "SD of event-level information content ({0})", "bits"),
    (r"^(chord|melody)_ic_p90$", "90th percentile of event information content ({0}); peak surprise", "bits"),
    (r"^(chord|melody)_entropy_mean$", "Mean entropy of the predictive distribution ({0}); uncertainty", "bits"),
    (r"^(chord|melody)_n_events$", "Number of {0} events scored", "count"),
]

CHORD_ALPHABET = [f"{d}{q}" for d in range(12) for q in ("M", "m")]
MELODY_ALPHABET = list(range(-12, 13))


class NGram:
    def __init__(self, alphabet, order=3, alpha=0.05, weights=None):
        self.alphabet = list(alphabet)
        self.index = {s: i for i, s in enumerate(self.alphabet)}
        self.V = len(self.alphabet)
        self.order = order
        self.alpha = alpha
        w = np.array(weights if weights is not None else [2.0 ** n for n in range(order)], dtype=float)
        self.w = w / w.sum()                         # weight for context length n = 0..order-1
        self.joint = [Counter() for _ in range(order)]
        self.ctx = [Counter() for _ in range(order)]

    def update(self, seq, sign=1):
        seq = [s for s in seq if s in self.index]
        for i, s in enumerate(seq):
            for n in range(self.order):
                if i - n < 0:
                    break
                c = tuple(seq[i - n:i])
                self.joint[n][(c, s)] += sign
                self.ctx[n][c] += sign

    def dist(self, context):
        p = np.zeros(self.V)
        for n in range(self.order):
            if len(context) < n:
                # back off: give this order's weight to the uniform distribution
                p += self.w[n] / self.V
                continue
            c = tuple(context[len(context) - n:]) if n else ()
            tot = self.ctx[n][c]
            counts = np.array([self.joint[n][(c, s)] for s in self.alphabet], dtype=float)
            p += self.w[n] * (counts + self.alpha) / (tot + self.alpha * self.V)
        return p / p.sum()

    def score(self, seq):
        seq = [s for s in seq if s in self.index]
        ic, h = [], []
        for i, s in enumerate(seq):
            p = self.dist(seq[max(0, i - self.order + 1):i])
            ic.append(-np.log2(p[self.index[s]]))
            h.append(float(-(p * np.log2(p)).sum()))
        return np.array(ic), np.array(h)


def corpus_surprise(sequences: dict[str, dict], cfg: dict) -> dict[str, dict]:
    """sequences: track_id -> {'chords_key_relative': [...], 'melody_intervals': [...]}."""
    results: dict[str, dict] = {tid: {} for tid in sequences}
    specs = [("chord", "chords_key_relative", CHORD_ALPHABET, int(cfg.get("chord_order", 3))),
             ("melody", "melody_intervals", MELODY_ALPHABET, int(cfg.get("melody_order", 3)))]
    for prefix, key, alphabet, order in specs:
        seqs = {t: s.get(key) for t, s in sequences.items() if s.get(key) and len(s.get(key)) >= 4}
        if len(seqs) < int(cfg.get("min_corpus_songs", 20)):
            continue
        model = NGram(alphabet, order=order, alpha=float(cfg.get("alpha", 0.05)))
        for s in seqs.values():
            model.update(s, +1)
        for tid, s in seqs.items():
            model.update(s, -1)                       # leave-one-out
            ic, h = model.score(s)
            model.update(s, +1)
            results[tid].update({
                f"{prefix}_ic_mean": float(ic.mean()), f"{prefix}_ic_sd": float(ic.std()),
                f"{prefix}_ic_p90": float(np.percentile(ic, 90)), f"{prefix}_entropy_mean": float(h.mean()),
                f"{prefix}_n_events": int(len(ic)),
            })
    return results
