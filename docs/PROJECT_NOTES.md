# Project notes: songfeatures pipeline

Hand-over notes for anyone (person or AI assistant) picking up this project. They record what the
pipeline does, the decisions taken and why, the bugs found and fixed, the test results, and what is
still open. Last updated: 25 October 2026.

## 1. Purpose

An open-source pipeline that extracts audio, structural and lyrical features from **full songs**
for a scientific research article on songs and their streaming performance. It reproduces
Spotify's 13 audio features (Spotify closed its Audio Features / Audio Analysis endpoints to new
apps in November 2024) and adds about 480 more.

- The study's songs are full-length files provided by the university. **They are copyrighted:
  never commit them, or results that contain their lyrics, to this public repository.**
- Testing uses 100 Creative Commons songs from the Free Music Archive (FMA), see `ci/fma_sample/`.

## 2. What the pipeline produces

`features.csv` (one row per song) + `codebook.csv` (every column: description, unit, tool,
reference). 13 modules, 62 feature groups, 498 features; the module-by-module overview with a
"provided by Spotify?" column is `docs/Song_Feature_Groups.xlsx`.

| Module | Features | What it covers |
|---|---|---|
| loudness | 15 | BS.1770 loudness, loudness range, peaks/compression, stereo, duration |
| timbre | 43 | spectral shape, 7 frequency bands, 13 MFCCs, timbral change |
| rhythm | 18 | tempo + stability, pulse, onsets, meter (Beat This!), syncopation, DFA danceability |
| harmony | 24 | key/mode, modulation, tuning, chords, harmonic rhythm, roughness |
| structure | 27 | sections, repetition, intro/chorus, energy arc, drops/build-ups, ending type |
| vocals | 30 | Demucs stems, vocal presence, sung vs spoken, pitch, intonation, vibrato, melody |
| voice_quality | 89 | openSMILE eGeMAPS on the isolated vocal |
| lyrics | 17 | Whisper transcription, vocabulary, repetitiveness, delivery rate, pronouns, sentiment |
| essentia_models | 156 | genre (Discogs 400), moods, valence/arousal over time, 56 mood/theme + 40 instrument tags |
| clap | 32 | zero-shot text-audio concepts (6 fixed reference + 10 configurable) |
| surprise (corpus) | 10 | chord and melody information content (own n-gram model, IDyOM-style) |
| novelty (corpus) | 21 | typicality / novelty from pretrained embeddings (Askin & Mauskapf 2017 style) |
| open_features | 16 | the 13 Spotify-equivalent `open_*` features + diagnostics |

Corpus modules compare each song with the others: compute them once on the full study sample.

## 3. How it runs

```bash
pip install -r requirements-core.txt -r requirements-models.txt && pip install -e .
python -m songfeatures download-models
python -m songfeatures run --audio-dir /path/to/songs --out results     # or --manifest songs.csv
# large sets: run shards in parallel, then merge
python -m songfeatures run --manifest songs.csv --out results --shard 3/20
python -m songfeatures merge --manifest songs.csv --out results
```
Other commands: `check`, `codebook`, `prepare-manifest`, `compare-spotify` (validate `open_*`
against Spotify values where available).

Internals worth knowing:
- Pass 1 (signal modules) runs in a process pool; Pass 2 runs each model module over all songs in
  its own fresh process (memory is returned to the OS between models); Pass 3 = corpus modules.
- Every song's results are cached in `results/cache/<track_id>.json`, so an interrupted run resumes.
- The vocal-activity mask is cached per song (`vocal_activity.npz`), so voice_quality and lyrics do
  not reload the voice model.

## 4. Key decisions (and why)

| Decision | Reason |
|---|---|
| Demucs `htdemucs` (single model) in CI; `htdemucs_ft` optional | Separation was 70% of run time; htdemucs is ~4x faster. On the 100 FMA songs, vocal features correlate r = 0.92-0.99 with htdemucs_ft (vocal presence 0.99, stem shares 0.98-0.99); `note_intonation_dev_cents` only 0.82, so use with caution or recompute with htdemucs_ft. Instrumental flag agrees on 98/100 songs |
| Whisper `large-v3`, beam size `auto` (5 with >= 12 GB RAM, else 1) | Beam 5 peaks at ~6 GB on CPU; beam 1 keeps 8 GB machines from running out of memory |
| Lyrics: Whisper gets the vocal passages found by the vocals module as `clip_timestamps`; Whisper's own VAD off | Whisper's VAD is a speech detector and discarded singing. A 12-song comparison of 4 set-ups: in-place clips recovered 2,458 words vs 1,836 (old) and 2,334 (concatenated passages) |
| Lyrics: keep a segment unless low-confidence, a known subtitle artefact, or "no speech" **and** word-sparse (< 1 word/s) | Whisper rates singing over music as "no speech"; the old `no_speech_prob` filter dropped whole verses of correct lyrics |
| Lyric language left empty when < 10 words | Whisper always guesses a language, even for humming |
| Meter: Beat This! downbeats, bar length from nearby downbeat pairs (`beats_per_bar` + `beats_per_bar_confidence`); `open_time_signature` uses it only if confidence >= 0.5, else the periodicity estimate | Beat This! (without DBN) marks spurious downbeats; the old median-count method gave 1 beat per bar for many songs |
| CLAP `laion/larger_clap_general`; analyse the `_margin` columns | `larger_clap_music` checkpoint is broken; the 0-1 probabilities saturate |
| Instrumentalness from source separation, with the Essentia voice model as a second opinion (`vocal_detectors_disagree`) | Each detector alone misfires on some genres |
| Novelty from Discogs-EffNet, MusiCNN and CLAP embeddings (cosine similarity) | Richer than the 11 Echo Nest features Askin & Mauskapf used |

## 5. Bugs found and fixed during testing

| Problem | Fix |
|---|---|
| Fade-outs misclassified; tuning wrapped at +/-50 cents; segmentation collapsed to one section; key mode filter too wide | ending classifier, tuning handling, segmentation guard + `structure_reliable` flag, narrower filter |
| Out-of-memory kill in the lyrics step | model steps in separate processes, cached vocal mask, memory-aware Whisper beam |
| Demucs ran on one CPU thread (thread setting leaked from Pass 1) | per-process thread initialiser |
| All GitHub jobs segfaulted in Pass 1 | librosa's numba cache was written by several workers at once on fresh machines; the cache is now warmed in the parent first |
| A worker crash aborted the whole shard | crashed songs are retried one per process and recorded as errors; the rest continue |
| Crashing jobs still showed "success" on GitHub | workflow uses explicit `bash` (pipefail) |
| `beats_per_bar` = 1 for many songs | robust downbeat-pair estimator (section 4) |
| 17 of 56 songs with vocals got 0 lyric words | lyrics method (section 4) |
| Lyric delivery rate (words/syllables per vocal second) always empty | computed from the cached vocal mask |
| Model download hung on a stalled server | 60 s timeout, 3 retries, `.part` files |

## 6. Test results (FMA 100-song sample, GitHub Actions)

Latest run (25 Oct 2026, htdemucs + new lyrics method): **no module failures on 100/100 songs**;
40 min for the whole run (20 parallel jobs), median job 26 min (previous settings: 72 / 52 min).

| Check against FMA labels | Result |
|---|---|
| FMA-flagged instrumental songs detected by separation | 9/12 (11/12 with htdemucs_ft) |
| Songs with vocals that got >= 10 lyric words | 54/58 (42/58 before the lyrics fix) |
| Whisper language = FMA tag | 17/22 (FMA tags are sometimes wrong, e.g. a Dutch song tagged English) |
| Essentia top genre = FMA genre | 45/100 top-1, 72/100 top-3 (FMA labels are artist-chosen and rough; chance ~ 7/100) |
| Clear Beat This! bar grid | 60/98; of these 58/60 agree with the periodicity estimate |

Time per song on a 4-core CPU, previous settings: separation 390 s (htdemucs_ft), lyrics 58 s,
rhythm 35 s, everything else ~30 s. With htdemucs: roughly 4-5 min per song. A GPU makes
separation and Whisper 10-50x faster; use one for the full study set.

Earlier test: 13 country songs from the university set, run locally; findings fed the fixes above.

## 7. Running the test on GitHub

- Workflow `.github/workflows/fma-sample-test.yml`: runs on every push to branch
  `ci/fma-sample-test` (or "Run workflow" in the Actions tab). Jobs: models (cached) -> 20 shards
  -> merge + report. Download the `fma-sample-results` artifact for `features.csv` and the report.
- Add `[skip ci]` to a commit message to push to that branch without starting a run.
- GitHub caches are per branch (plus the default branch): a new branch re-downloads ~5 GB of models.
- `.github/workflows/lyrics-probe.yml`: diagnostic comparison of Whisper set-ups, manual only:
  `gh workflow run lyrics-probe.yml --ref ci/fma-sample-test` (reuses that branch's model cache).
- Public repo = free 4-core / 16 GB runners and unlimited minutes.

## 8. Known limitations

- `open_*` features are open equivalents of Spotify's, not Spotify's own numbers; validate on songs
  where Spotify values exist (`compare-spotify`).
- Pronoun and sentiment lyric features are English only; Whisper can misdetect language on old or
  unusual recordings (an old Russian recording was detected as Latin).
- `note_intonation_dev_cents` is sensitive to the separation model.
- Corpus features (surprise, novelty) depend on the songs analysed together.
- Novelty's period window is +/-1 year and includes later songs; Askin & Mauskapf used the past
  52 weeks only, genre-weighted (see next steps).

## 9. Next steps

- [ ] Album-relative atypicality: leave-one-out distance from the album average, separately for
      sonic, structural and lyrical features, plus album composition measures.
- [ ] Askin & Mauskapf replication variant: past-52-weeks, genre-weighted typicality from the
      `open_*` features and from embeddings.
- [ ] Remove the stale `build/` folder; update the README "Runtime" section with measured times.
- [ ] Delete leftover branches `ci/debug-shard` and `ci/lyrics-probe`.
- [ ] Run the full study set on a GPU machine (university cluster or similar).
- [ ] Validate `open_*` against Spotify values for a subset of songs.

## 10. Files in `docs/`

- `PROJECT_NOTES.md`: this file.
- `Song_Feature_Groups.xlsx`: every module's feature groups, counts, explanations, Spotify
  availability and code links.
- `fma_test_report.md`: report of the latest 100-song FMA run.
