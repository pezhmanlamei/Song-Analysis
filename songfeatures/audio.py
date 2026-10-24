"""Audio loading and the per-track analysis context.

A `Track` decodes the file once and lazily computes shared intermediate
representations (STFT, HPSS, onset envelope, beats, chroma, ...) so that every
feature module re-uses them instead of recomputing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

SR_HI = 44100          # analysis rate for loudness, production and separation
SR = 22050             # analysis rate for librosa-based MIR features
N_FFT = 2048
HOP = 512


class AudioLoadError(RuntimeError):
    pass


def _ffprobe_channels(path: Path) -> int | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=channels", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=60).stdout
        return int(json.loads(out)["streams"][0]["channels"])
    except Exception:
        return None


def decode_audio(path: str | Path, sr: int = SR_HI) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 array shaped (channels, samples).

    Mono files stay mono (1 channel); anything with >=2 channels is down-mixed to stereo.
    """
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"file not found: {path}")
    if shutil.which("ffmpeg"):
        n_in = _ffprobe_channels(path)
        ch = 1 if n_in == 1 else 2
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
               "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(sr), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise AudioLoadError(proc.stderr.decode(errors="ignore")[-500:])
        data = np.frombuffer(proc.stdout, dtype=np.float32)
        if data.size == 0:
            raise AudioLoadError("decoded zero samples")
        return data.reshape(-1, ch).T.copy()
    # Fallback without ffmpeg (wav/flac/ogg/mp3 via libsndfile)
    import soundfile as sf
    import librosa
    data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    data = data.T
    if data.shape[0] > 2:
        data = data[:2]
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr, axis=-1)
    return np.ascontiguousarray(data, dtype=np.float32)


_BEAT_THIS = None


def _beat_this(y: np.ndarray, sr: int, device: str = "auto"):
    """Neural beat/downbeat tracking with Beat This! (CPJKU, MIT licence) on an already
    decoded mono signal (avoids torchaudio's file loading, which lacks MP3 support)."""
    global _BEAT_THIS
    from beat_this.inference import Audio2Beats
    if _BEAT_THIS is None:
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _BEAT_THIS = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = _BEAT_THIS(np.asarray(y, dtype=np.float64), sr)
    return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


def hpcp_frames(y_hi: np.ndarray, tuning_cents: float = 0.0, harmonics: int = 4) -> np.ndarray:
    """Essentia HPCP (Gomez 2006) frames on the 44.1 kHz mono mix, hop 1024 so the frame
    grid equals librosa's 512-hop grid at 22.05 kHz. Returned rows are C..B."""
    import essentia.standard as es
    w = es.Windowing(type="blackmanharris62")
    spec = es.Spectrum()
    peaks = es.SpectralPeaks(orderBy="magnitude", magnitudeThreshold=1e-5, minFrequency=20,
                             maxFrequency=3500, maxPeaks=60, sampleRate=SR_HI)
    white = es.SpectralWhitening(maxFrequency=3500, sampleRate=SR_HI)
    hpcp = es.HPCP(size=12, referenceFrequency=440.0 * 2 ** (tuning_cents / 1200), harmonics=harmonics,
                   bandPreset=True, minFrequency=20, maxFrequency=3500, weightType="cosine",
                   nonLinear=False, windowSize=1.0, normalized="none", sampleRate=SR_HI)
    frames = []
    for fr in es.FrameGenerator(np.ascontiguousarray(y_hi, dtype=np.float32), frameSize=4096,
                                hopSize=1024, startFromZero=False):
        s = spec(w(fr))
        f, m = peaks(s)
        frames.append(hpcp(f, white(s, f, m)))
    H = np.array(frames, dtype=np.float64).T
    return np.roll(H, -3, axis=0)       # Essentia bin 0 = A -> rotate so row 0 = C


class Track:
    """All audio representations of one song, computed lazily and cached in memory."""

    def __init__(self, track_id: str, path: str | Path, meta: dict | None = None,
                 cfg: dict | None = None, artifact_dir: Path | None = None):
        self.track_id = track_id
        self.path = Path(path)
        self.meta = meta or {}
        self.cfg = cfg or {}
        self.artifact_dir = artifact_dir
        self._cache: dict[str, Any] = {}
        # Outputs that are not scalar features
        self.timeseries: dict[str, np.ndarray] = {}
        self.sequences: dict[str, list] = {}
        self.embeddings: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ cache
    def get(self, key: str, fn: Callable[[], Any]):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def put(self, key: str, value: Any):
        self._cache[key] = value

    def has(self, key: str) -> bool:
        return key in self._cache

    # ------------------------------------------------------------------ audio
    @property
    def raw(self) -> np.ndarray:
        """(channels, n) float32 at 44.1 kHz; channels is 1 for mono files."""
        def _load():
            a = decode_audio(self.path, SR_HI)
            min_s = float(self.cfg.get("min_duration_s", 10.0))
            if a.shape[1] < min_s * SR_HI:
                raise AudioLoadError(f"track shorter than {min_s}s")
            if float(np.max(np.abs(a))) < 1e-6:
                raise AudioLoadError("track is silent")
            return a
        return self.get("raw", _load)

    @property
    def stereo(self) -> np.ndarray:
        """(2, n) at 44.1 kHz (mono duplicated)."""
        r = self.raw
        return r if r.shape[0] == 2 else np.repeat(r, 2, axis=0)

    @property
    def is_mono(self) -> bool:
        return self.raw.shape[0] == 1

    @property
    def y_hi(self) -> np.ndarray:
        """Mono mix at 44.1 kHz."""
        return self.get("y_hi", lambda: self.raw.mean(axis=0).astype(np.float32))

    @property
    def y(self) -> np.ndarray:
        """Mono mix at 22.05 kHz."""
        import librosa
        return self.get("y", lambda: librosa.resample(self.y_hi, orig_sr=SR_HI, target_sr=SR))

    def y_at(self, sr: int) -> np.ndarray:
        import librosa
        return self.get(f"y_{sr}", lambda: librosa.resample(self.y_hi, orig_sr=SR_HI, target_sr=sr))

    @property
    def duration(self) -> float:
        return self.raw.shape[1] / SR_HI

    # ---------------------------------------------------------- spectral reps
    @property
    def D(self) -> np.ndarray:
        import librosa
        return self.get("D", lambda: librosa.stft(self.y, n_fft=N_FFT, hop_length=HOP))

    @property
    def S(self) -> np.ndarray:
        return self.get("S", lambda: np.abs(self.D))

    @property
    def hpss(self) -> tuple[np.ndarray, np.ndarray]:
        """Complex harmonic and percussive STFTs (Fitzgerald median-filtering HPSS)."""
        import librosa
        return self.get("hpss", lambda: librosa.decompose.hpss(self.D))

    @property
    def y_harm(self) -> np.ndarray:
        import librosa
        return self.get("y_harm", lambda: librosa.istft(self.hpss[0], hop_length=HOP, length=len(self.y)))

    @property
    def onset_env(self) -> np.ndarray:
        import librosa
        return self.get("onset_env", lambda: librosa.onset.onset_strength(y=self.y, sr=SR, hop_length=HOP))

    @property
    def beats(self) -> dict:
        """Beat tracking. Default: librosa (Ellis 2007 dynamic programming).
        cfg beat_tracker: beat_this -> neural beats + downbeats (Foscarin et al. 2024)."""
        import librosa

        def _beats():
            downbeats = None
            backend = self.cfg.get("beat_tracker", "librosa")
            times = None
            if backend == "beat_this":
                try:
                    times, downbeats = _beat_this(self.y_hi, SR_HI, self.cfg.get("device", "auto"))
                    self.put("beat_backend", "beat_this")
                except ImportError:
                    times = None
            if times is None or len(times) < 8:
                self.put("beat_backend", "librosa")
                tempo, frames = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=SR,
                                                        hop_length=HOP, units="frames")
                tempo = float(np.atleast_1d(tempo)[0])
                frames = np.asarray(frames, dtype=int)
                if frames.size < 8:  # fallback: regular grid at the estimated tempo
                    step = 60.0 / max(tempo, 60.0)
                    frames = librosa.time_to_frames(np.arange(0.0, self.duration, step), sr=SR, hop_length=HOP)
                times = librosa.frames_to_time(frames, sr=SR, hop_length=HOP)
            times = np.asarray(times, dtype=np.float64)
            frames = librosa.time_to_frames(times, sr=SR, hop_length=HOP)
            frames, keep = np.unique(frames, return_index=True)
            times = times[keep]
            ibi = np.diff(times)
            tempo = float(60.0 / np.median(ibi)) if ibi.size else float("nan")
            return {"tempo": tempo, "frames": frames, "times": times, "downbeats": downbeats}
        return self.get("beats", _beats)

    @property
    def chroma(self) -> np.ndarray:
        """Un-normalised CQT chroma (12 x frames) of the full mix (default) or the HPSS
        harmonic component (cfg chroma_source: harmonic). The full mix is more robust for
        short chord stabs/strums, which HPSS tends to assign to the percussive layer."""
        import librosa

        def _chroma():
            y = self.y_harm if self.cfg.get("chroma_source", "mix") == "harmonic" else self.y
            tuning = None
            try:   # Essentia's dedicated estimator is more stable near the +/-50-cent boundary
                import essentia.standard as es
                tf = np.asarray(es.TuningFrequencyExtractor()(np.ascontiguousarray(self.y_hi, dtype=np.float32)))
                tf = tf[(tf > 400) & (tf < 480)]
                if tf.size:
                    tuning = float(np.clip(np.log2(np.median(tf) / 440.0) * 12, -0.5, 0.5))
                    self.put("tuning_method", "essentia")
            except ImportError:
                pass
            if tuning is None:
                tuning = float(librosa.estimate_tuning(y=y, sr=SR))
                self.put("tuning_method", "librosa")
            self.put("tuning", float(tuning))
            n_frames = 1 + len(self.y) // HOP
            method = self.cfg.get("chroma_method", "auto")
            if method in ("auto", "hpcp"):
                try:
                    H = hpcp_frames(self.y_hi, tuning_cents=100 * tuning)
                    self.put("chroma_method", "essentia_hpcp")
                    if H.shape[1] < n_frames:
                        H = np.pad(H, ((0, 0), (0, n_frames - H.shape[1])))
                    return H[:, :n_frames]
                except ImportError:
                    if method == "hpcp":
                        raise
            self.put("chroma_method", "librosa_cqt")
            return librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, tuning=tuning, norm=None)
        return self.get("chroma", _chroma)

    @property
    def tuning(self) -> float:
        _ = self.chroma
        return self._cache["tuning"]

    @property
    def mfcc(self) -> np.ndarray:
        import librosa
        return self.get("mfcc", lambda: librosa.feature.mfcc(y=self.y, sr=SR, n_mfcc=20,
                                                            n_fft=N_FFT, hop_length=HOP))

    def beat_sync(self, X: np.ndarray, aggregate=np.median) -> np.ndarray:
        import librosa
        return librosa.util.sync(X, self.beats["frames"], aggregate=aggregate)

    # --------------------------------------------------------------- outputs
    def save_artifacts(self):
        """Persist time series, symbolic sequences and embeddings next to the cache."""
        if self.artifact_dir is None:
            return
        d = Path(self.artifact_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.timeseries:
            f = d / "timeseries.npz"
            old = dict(np.load(f, allow_pickle=False)) if f.exists() else {}
            old.update({k: np.asarray(v) for k, v in self.timeseries.items()})
            np.savez_compressed(f, **old)
        if self.sequences:
            f = d / "sequences.json"
            old = json.loads(f.read_text()) if f.exists() else {}
            old.update(self.sequences)
            f.write_text(json.dumps(old))
        if self.embeddings:
            f = d / "embeddings.npz"
            old = dict(np.load(f, allow_pickle=False)) if f.exists() else {}
            old.update({k: np.asarray(v, dtype=np.float32) for k, v in self.embeddings.items()})
            np.savez_compressed(f, **old)
