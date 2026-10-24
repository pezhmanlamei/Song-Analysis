# FMA 100-song test sample

100 full-length songs from the Free Music Archive (FMA), 10 per genre: Blues, Classical, Country,
Electronic, Folk, Hip-Hop, Jazz, Pop, Rock, Soul-RnB. Selection: unambiguous FMA top-level genre,
1.5-8 minutes, at most 2 songs per artist, drawn from 3-5 parts of the archive per genre;
6.2 hours of audio, released 2008-2016.

* `manifest.csv` — pipeline input. `fma_shard` / `fma_row` locate each song in the Hugging Face
  copy of FMA-full (`benjamin-paine/free-music-archive-full`); `sha256` verifies the download.
  Extra columns for validation: `fma_listens` (play count), `fma_language`,
  `fma_instrumental_flag` (1 = flagged instrumental; 0 = not flagged), `genre_labels`.
* `ATTRIBUTION_AND_LICENSES.csv` — artist, title and licence of every song (Creative Commons and
  similar free licences, mostly non-commercial). Credit the artists if you publish derived material.

Download the audio: `python ci/fetch_fma_sample.py --manifest ci/fma_sample/manifest.csv`
