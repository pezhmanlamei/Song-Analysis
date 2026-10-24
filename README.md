# songfeatures — open-source audio features for full-length songs

A reproducible pipeline that turns a folder (or CSV manifest) of full songs into one row
of ~450 interpretable features per song, plus a **codebook** documenting every feature
(definition, unit, tool, reference) for your methods section. Everything is open source;
no Spotify or other proprietary API is used.

It measures what Spotify's 12 "audio features" do not: how a song **unfolds over time**
(structure, energy arc, drops, fades, emotion trajectory), **harmony** (chords, key
changes, harmonic complexity), **the voice** (presence, pitch, intonation/pitch-correction,
vibrato, voice quality, lyrics), **production** (loudness war, dynamics, stereo, frequency
balance), and how a song relates to **the rest of the market** (harmonic/melodic surprise,
novelty vs contemporaries, genre typicality, artist novelty).

## Feature families

| Module | What it measures (examples) | Main tool(s) |
|---|---|---|
| `loudness` | integrated loudness (LUFS), loudness range, true peak, peak-to-loudness ratio, crest factor, clipping, stereo width/correlation | pyloudnorm (ITU-R BS.1770-4), own EBU Tech 3342 |
| `timbre` | brightness, flatness, spectral flux/contrast/tilt, energy share in 7 frequency bands, 13 MFCCs, timbral change rate | librosa, scipy |
| `rhythm` | tempo, tempo stability/drift, pulse clarity, beat strength, onset density, IOI entropy, syncopation (off-beat onset share), percussiveness, DFA danceability, meter (with Beat This!) | librosa, Essentia, optional Beat This! |
| `harmony` | key, key clarity/ambiguity, continuous majorness, local keys, modulations, late "truck-driver" key change, chords (C, Am, …), harmonic rhythm, chord vocabulary/entropy, root motion, triadicity, tonal change, roughness, tuning | Essentia HPCP + KeyExtractor, librosa, own HMM chord/key models |
| `structure` *(full song only)* | number/types of sections, structural entropy, repetition, intro length, hook-first, time to first chorus, chorus share/repeats/loudness contrast, energy slope/peak/arc, drops, build-ups, silences, fade-out | Laplacian segmentation (McFee & Ellis 2014); optional allin1 |
| `vocals` | stem balance (vocals/drums/bass/other), vocal presence, time to first vocal, phrases, f0 median/range, intonation & pitch-correction indices, vibrato share/rate/extent, melodic intervals (steps/leaps/repeats) | Demucs htdemucs, librosa pYIN |
| `voice_quality` | 88 eGeMAPS voice parameters of the lead vocal (jitter, shimmer, HNR, formants, spectral slope, …) | openSMILE |
| `lyrics` | language, word count, lexical diversity (MATTR), repetitiveness (compression, repeated lines), delivery rate, pronouns, sentiment (English) | faster-whisper, VADER |
| `essentia_models` | valence & arousal **time series** (2 models), happy/sad/aggressive/relaxed/party/acoustic/electronic, danceability, voice, tonality, approachability, engagement, 56 mood/theme tags, 40 instruments, 400 Discogs styles + 15 parent genres | Essentia pretrained models |
| `clap` | zero-shot scores for constructs **you define in words** ("nostalgic", "cinematic", …) | LAION CLAP |
| `surprise` *(corpus)* | harmonic and melodic surprise (information content) and uncertainty (entropy), leave-one-out vs your corpus | own n-gram model (IDyOM logic) |
| `novelty` *(corpus)* | typicality vs whole corpus / contemporaries (±1 year) / same genre, nearest-neighbour uniqueness, novelty vs the artist's earlier songs | embeddings (Discogs-EffNet, CLAP) |

The complete list with definitions is in `codebook_full.csv` (or `python -m songfeatures codebook`).

## Spotify-equivalent features (`open_*`)

Every run also writes 13 columns named after Spotify's audio features, on the same scales.
They are **open measurements of the same constructs, not Spotify's values** (Spotify's
algorithms are proprietary), so do not mix them with Spotify-API numbers in one variable.
If you need comparability with earlier Spotify-based studies, compute both on an overlap
sample and report their correlation. Each song's `open_sources` column records which
source produced each value.

| Column | Primary source | Fallback |
|---|---|---|
| `open_duration_ms` | decoded audio length | — |
| `open_tempo` | `tempo_bpm` | — |
| `open_key` (0-11, -1 none) / `open_mode` (1/0) | Essentia KeyExtractor | key-profile estimate |
| `open_loudness` | integrated loudness, LUFS (ITU-R BS.1770) | — |
| `open_time_signature` | Beat This! downbeats | beat-periodicity estimate (3 or 4) |
| `open_danceability` | Essentia danceability classifier | DFA danceability / 3 |
| `open_energy` | mean of signal composite* and model arousal | signal composite only |
| `open_valence` | DEAM valence, rescaled 1-9 → 0-1 | emoMusic, then CLAP happy-vs-sad |
| `open_acousticness` | Essentia acoustic-mood probability | CLAP acoustic-vs-electronic |
| `open_instrumentalness` | 1 − vocal presence / 0.2 (separated vocals) | 1 − Essentia voice probability |
| `open_speechiness` | spoken/rapped vocal seconds ÷ duration | CLAP spoken-vs-sung |
| `open_liveness` | CLAP live-with-audience vs studio | — (NaN without CLAP) |

\* Signal energy = mean of five components mapped to 0-1 with fixed ranges: loudness
(−30…−5 LUFS), onset rate (0…8/s), energy above 2 kHz (0…40%), inverse loudness range
(20…0 LU), spectral flatness (0…0.3). Absolute values run lower than Spotify's typical
energy numbers; use them relatively (z-scores) or calibrate on an overlap sample.
Speechiness and liveness are heuristic / zero-shot measures: validate them before relying
on them (speech detection was checked on real LibriSpeech speech: 93% speech-like; and on
sung vocals: 0%).

## Install

Python 3.9-3.13, Linux or macOS recommended (Essentia has no Windows wheels; on Windows use WSL).

```bash
python -m venv ~/venvs/songfeatures && source ~/venvs/songfeatures/bin/activate
pip install -r requirements-core.txt
pip install essentia-tensorflow           # HPCP/key/danceability + pretrained models
# deep-learning modules (GPU strongly recommended): install torch for your CUDA first
pip install -r requirements-models.txt
# ffmpeg must be on PATH (conda install ffmpeg / apt install ffmpeg / module load ffmpeg)

python -m songfeatures check              # shows what is installed
python -m songfeatures download-models    # Essentia models (~300 MB) + Demucs/Whisper/CLAP weights
python tests/run_tests.py                 # 41 ground-truth checks (synthetic songs + real recordings when online)
```

## Run

```bash
# simplest: every audio file in a folder (track_id = file name)
python -m songfeatures run --audio-dir /data/songs --out results --workers 8

# recommended: a manifest with metadata (needed for novelty vs contemporaries/artist)
python -m songfeatures run --manifest manifest.csv --out results --config config.yaml --workers 8

# pilot on 20 songs, or only some modules
python -m songfeatures run --manifest manifest.csv --out pilot --limit 20
python -m songfeatures run --manifest manifest.csv --out results --modules loudness,rhythm,harmony
```

**From a track list (e.g. a Kaggle/Spotify CSV or XLSX) to a manifest:**

```bash
# match track_id/artists/track_name rows to your audio files (Spotify ID in the file name or
# path, else artist + title from file names / tags); writes manifest.csv plus
# manifest_unmatched.csv and manifest_unused_files.csv for checking
python -m songfeatures prepare-manifest --dataset dataset.xlsx --audio-dir /data/songs --out manifest.csv
#   --dedupe artist_title   collapse alternate versions (remasters, live, ...) of the same song
#   --pilot 50              random sample of 50 matched songs for a first run
```

**Manifest columns** (CSV): `audio_path` (required), `track_id` (e.g. your Spotify ID),
and optionally `artist`, `title`, `release_date` or `year`, `genre`, `lyrics_path`
(plain-text lyrics; more reliable than transcription), `vocals_path` (if you already have
isolated vocals). Any other columns (e.g. streams, chart position) are carried into the
output unchanged. See `examples/manifest_example.csv`.

**Large datasets: shard across cluster jobs.** `--shard K/N` processes every N-th song; run N
jobs (e.g. a SLURM array, see `examples/slurm_job.sh`) into the same `--out` folder, then
`python -m songfeatures merge --manifest manifest.csv --out results --config config.yaml`
builds the corpus features and the merged tables (identical to a single run).

**Validate against Spotify** when you have Spotify's audio features for the same track IDs
(e.g. the full Kaggle "Spotify Tracks" file):
`python -m songfeatures compare-spotify --features results/features.csv --spotify spotify.csv`
writes per-feature n, Pearson r, Spearman rho, means, and agreement for key/mode/time
signature (tempo agreement tolerates double/half-tempo errors).

The run is **resumable**: results are cached per song and module, so re-running the same
command after an interruption (or a cluster time limit) continues where it stopped.
Modules whose packages/models are missing are skipped once with an install hint.
See `examples/slurm_job.sh` for a university GPU cluster.

## Outputs (`--out` folder)

| File | Content |
|---|---|
| `features.csv` | one row per song: manifest columns + all features |
| `codebook.csv` | definition, unit, module, tool and reference of every column in `features.csv` |
| `errors.csv` | any module that failed for any song, with the error message |
| `run_info.json` | effective configuration, library versions, timing — for reproducibility |
| `corpus_features.csv` | surprise and novelty features |
| `artifacts/<id>/timeseries.npz` | curves for plots/dynamic analyses: short-term loudness, valence/arousal, vocal f0, local key |
| `artifacts/<id>/sequences.json` | beats, sections, chord sequence (absolute and key-relative), vocal notes/intervals |
| `artifacts/<id>/embeddings.npz` | song embeddings (Discogs-EffNet, MusiCNN, CLAP) |
| `artifacts/<id>/stems/vocals.flac` | separated vocal stem (configurable) |

## Runtime (rough)

On one CPU core a 4-minute song needs ~30 s for the signal-processing modules and ~15 s for
pitch/voice features. With a GPU, Demucs + Whisper + CLAP add roughly 20-40 s per song; the
Essentia models ~10 s on CPU. Ballpark: **1,000 songs ≈ 1 h for the core pass with 8 workers
and ≈ 8-15 h for the model pass on one GPU.** For ~80,000 songs that is roughly 1-2 GPU-months
in total, so shard it (e.g. 40 GPUs ≈ 1-2 days), and consider cheaper settings: Whisper
`small` instead of `large-v3` (or provided lyrics), `demucs_shifts: 0`. Run a 50-song
pilot first to time your setup.

## 100-song test on GitHub Actions

`.github/workflows/fma-sample-test.yml` runs the whole pipeline on 100 Creative Commons songs
from the Free Music Archive (10 genres x 10; listed with licences in
`ci/fma_sample/ATTRIBUTION_AND_LICENSES.csv`). No audio is stored in the repository: each job
downloads its songs from the FMA dataset on Hugging Face and verifies them by SHA-256
(`ci/fetch_fma_sample.py`).

* **Start it:** Actions tab → "FMA 100-song pipeline test" → Run workflow (or push to the
  `ci/fma-sample-test` branch).
* **How it runs:** one job installs and caches the model weights, 20 parallel jobs process 5
  songs each (`--shard K/20`), and a merge job builds `features.csv`, the corpus features and a
  report (`ci/report_fma_sample.py`) that also checks the output against FMA's own labels
  (instrumental flag, language, genre, live albums).
* **Results:** the run summary page shows the report; download the `fma-sample-results`
  artifact for `features.csv`, `codebook.csv`, `errors.csv` and `report/`.
* **Settings:** `ci/config_ci.yaml` (Demucs `htdemucs_ft`, Whisper `large-v3`, CLAP
  `larger_clap_general`, Beat This! for `beats_per_bar`); pinned versions in
  `ci/requirements-ci.txt`.
* **Cost:** free on standard runners in a public repository. In a private repository the
  run uses roughly 2,000-3,000 of the plan's monthly Actions minutes (2 CPUs per job there).

Do not use GitHub Actions for licensed/copyrighted audio (e.g. a university's song collection):
keep that on infrastructure covered by the licence.

## Methodological notes (please read before publishing)

* **Validate.** Automatic features are estimates. Hand-annotate a random subsample
  (e.g. 100 songs: key, first-chorus time, fade-out, vibrato, a few CLAP constructs) and
  report agreement. `tests/run_tests.py` checks the code on synthetic ground truth only.
* **Heuristics are labelled as such** in the codebook: the chorus is the loudest repeated
  section type (or allin1's learned label); drops/build-ups are loudness rules; syncopation is
  the off-beat onset share and depends on correct beat phase (use `beat_tracker: beat_this`
  for reggae/funk/afrobeat-heavy samples).
* **Vocal features depend on separation quality.** Inspect a few `stems/vocals.flac`.
  Whisper can hallucinate on singing; prefer `lyrics_path` when you have lyrics.
* **Many features are correlated.** Plan a theory-driven subset or a factor analysis / PCA.
* **Corpus features** (surprise, novelty) are relative to the songs you analyse; they need
  ≥ 20 songs (configurable) and are best with hundreds.
* **Licences.** openSMILE and the Essentia model weights (CC BY-NC-SA 4.0) are free for
  **non-commercial research** only; the rest (librosa, Essentia library under AGPL, Demucs,
  Whisper, CLAP, pyloudnorm) are open-source licences that permit research use. Keep audio and lyric
  transcripts private; share features, not content.
* **Report versions**: cite the tools listed in `codebook.csv` and attach `run_info.json`.

## Layout

```
songfeatures/
  audio.py            decoding (ffmpeg) + Track: cached STFT, HPSS, beats, chroma/HPCP, MFCC
  features/           loudness, timbre, rhythm, harmony, structure, vocals, voice_quality,
                      lyrics, essentia_models, clap
  corpus/             surprise (n-gram IC/entropy), novelty (embedding typicality)
  pipeline.py         batch runner (parallel, cached, resumable), preflight, merging
  codebook.py         feature documentation
config.yaml           all settings with comments
tests/                synthetic ground-truth songs + validation suite
examples/             manifest template, SLURM job script
```
