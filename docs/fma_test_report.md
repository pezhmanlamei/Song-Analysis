# FMA 100-song pipeline test

Songs in manifest: 100; rows in features.csv: 100; columns: 514

| module | songs ok |
|---|---|
| loudness | 100/100 |
| timbre | 100/100 |
| rhythm | 100/100 |
| harmony | 100/100 |
| structure | 100/100 |
| vocals | 100/100 |
| voice_quality | 100/100 |
| lyrics | 100/100 |
| essentia_models | 100/100 |
| clap | 100/100 |

No module failures.

Feature columns empty for every song: 2 (meter_backend_error, essentia_missing_heads)

Label checks below use the 100 processed songs.

## Checks against FMA labels

| check | result |
|---|---|
| FMA-flagged instrumental songs detected as instrumental by separation | 9/12 |
| ... and by the Essentia voice model (p < 0.5) | 8/12 |
| Whisper language = FMA language tag (songs with both) | 17/22 |
| Essentia top genre matches the FMA genre | 45/100 |
| ... per FMA genre | Blues 20%, Classical 60%, Country 30%, Electronic 100%, Folk 60%, Hip-Hop 60%, Jazz 50%, Pop 0%, Rock 70%, Soul-RnB 0% |
| CLAP live-vs-studio margin, albums titled 'Live' vs others (higher = more live) | -0.208 (n=23) vs -0.249 (n=77) |
| Meter: songs with a clear Beat This! bar grid (confidence >= 0.5; else the periodicity estimate is used) | 60/98 |
| ... of these, Spotify-scale meter = periodicity estimate | 58/60 |
| Songs where separation and the voice model disagree on vocals (listen to these) | 20/100 |

FMA's instrumental flag is only reliable when set to 1 (0 means 'not flagged').

## Median of selected features by genre

| genre      |   tempo_bpm |   loudness_integrated_lufs |   open_energy |   open_danceability |   open_valence |   open_acousticness |   open_instrumentalness |   open_speechiness |   vocal_presence_ratio |   chord_change_rate_per_min |   n_sections |
|:-----------|------------:|---------------------------:|--------------:|--------------------:|---------------:|--------------------:|------------------------:|-------------------:|-----------------------:|----------------------------:|-------------:|
| Blues      |      134.32 |                     -18.67 |          0.37 |                0.12 |           0.51 |                0.76 |                    0.01 |               0.06 |                   0.2  |                       12.5  |          6.5 |
| Classical  |      118.81 |                     -22.18 |          0.33 |                0.05 |           0.39 |                0.88 |                    0.96 |               0    |                   0.01 |                       18.74 |         11.5 |
| Country    |      144.14 |                     -16.12 |          0.46 |                0.03 |           0.49 |                0.86 |                    0    |               0.36 |                   0.46 |                       20.89 |         11   |
| Electronic |      130.84 |                     -10.17 |          0.48 |                0.92 |           0.57 |                0.1  |                    0.93 |               0    |                   0.01 |                        7.87 |         10.5 |
| Folk       |      131.8  |                     -11.07 |          0.45 |                0.1  |           0.54 |                0.81 |                    0    |               0.12 |                   0.32 |                       16.63 |         10   |
| Hip-Hop    |      104.62 |                      -9.46 |          0.51 |                0.94 |           0.55 |                0.05 |                    0    |               0.55 |                   0.57 |                        3.73 |          7   |
| Jazz       |      127.27 |                     -20.96 |          0.35 |                0.18 |           0.45 |                0.65 |                    1    |               0    |                   0    |                       13.78 |         14   |
| Pop        |      129.22 |                     -17.47 |          0.41 |                0.18 |           0.44 |                0.49 |                    0    |               0.3  |                   0.39 |                       18.34 |          8   |
| Rock       |      118.98 |                     -10.43 |          0.47 |                0.42 |           0.59 |                0.2  |                    0    |               0.21 |                   0.25 |                        8.28 |          9.5 |
| Soul-RnB   |      111.01 |                      -8.49 |          0.5  |                0.97 |           0.65 |                0.08 |                    0.84 |               0    |                   0.03 |                        8.74 |          6   |
