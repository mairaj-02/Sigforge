# SigForge

A modular, CLI-based signal processing toolkit for audio, image, and video files. Built on NumPy, SciPy, OpenCV, librosa, and moviepy — designed around a registry pattern that makes adding new operations a two-step process with zero changes to the control flow.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Architecture — The Registry Pattern](#architecture--the-registry-pattern)
5. [Audio Module](#audio-module)
   - [Fourier Analysis (FFT)](#fourier-analysis-fft)
   - [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
   - [IIR Filter Design and Application](#iir-filter-design-and-application)
   - [FIR Filter Design](#fir-filter-design)
   - [Pitch Detection](#pitch-detection)
   - [Noise Reduction](#noise-reduction)
6. [Image Module](#image-module)
   - [Spatial Filtering](#spatial-filtering)
   - [2D Frequency Domain Analysis](#2d-frequency-domain-analysis)
   - [Colour Space Conversion](#colour-space-conversion)
   - [Morphological Operations](#morphological-operations)
7. [Video Module](#video-module)
   - [Temporal Operations](#temporal-operations)
   - [Optical Flow](#optical-flow)
   - [Scene Change Detection](#scene-change-detection)
8. [Utility Modules](#utility-modules)
9. [Testing](#testing)
10. [Development Phases](#development-phases)
11. [Library Stack](#library-stack)

---

## Overview

SigForge processes three media types through a unified CLI:

- **Audio** — FFT, STFT, spectrograms, IIR/FIR filtering, pitch detection, spectral denoising
- **Image** — Gaussian blur, Sobel/Canny edge detection, 2D FFT, morphological ops, colour space conversion `will be implemented later` 
- **Video** — frame extraction, temporal averaging, optical flow (Lucas-Kanade / Farneback), scene change detection `will be implemented later`

The entry point is `main.py`, which routes through `cli.py` (Typer) → `detector.py` (file-type detection) → `menu.py` (InquirerPy operation selector) → the appropriate module handler. Each module exposes a `registry.py` that maps operation names to functions. Adding a new operation means writing the function and adding one line to the registry — no other file changes required.

---

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run interactively (prompts for operation and parameters)
sigforge audio.wav

# Run with flags (non-interactive)
sigforge audio.wav --op lowpass --cutoff 1000 --order 4 --verbose

# Process an image
sigforge photo.png --op sobel

# Process a video
sigforge clip.mp4 --op optical_flow_lk
```

---

## Project Structure

```
sigforge/
├── main.py                 # Entry point — parses CLI args, calls detector, hands off to handler
├── cli.py                  # Typer CLI definitions — @app.command(), options, arguments
├── detector.py             # MIME/extension detection → returns "audio" | "image" | "video"
├── menu.py                 # InquirerPy interactive menus — operation selector, parameter prompts
│
├── audio/
│   ├── __init__.py
│   ├── handler.py          # Orchestrator — loads audio, calls registry[op], saves output
│   ├── io.py               # soundfile / pydub read-write wrappers
│   ├── spectrum.py         # FFT, STFT, spectrogram computation
│   ├── filters.py          # IIR (Butterworth, Chebyshev, Elliptic) and FIR filter design & application
│   ├── pitch.py            # Pitch detection — autocorrelation method and YIN (via librosa)
│   ├── denoise.py          # Spectral subtraction and Wiener-style noise reduction
│   └── registry.py         # {"fft": compute_fft, "lowpass": apply_lowpass, ...}
│
├── image/
│   ├── __init__.py
│   ├── handler.py          # Orchestrator — loads image, calls registry[op], saves output
│   ├── io.py               # PIL / OpenCV read-write wrappers
│   ├── spatial.py          # Gaussian blur, Sobel edge detection, Canny edge detection
│   ├── frequency.py        # 2D FFT, magnitude spectrum, ideal frequency-domain LPF/HPF
│   ├── colour.py           # Colour space conversions — BGR↔RGB↔HSV↔YCrCb↔Grayscale
│   ├── morphology.py       # Erosion, dilation, opening, closing with configurable structuring elements
│   └── registry.py         # {"gaussian_blur": gaussian_blur, "sobel": apply_sobel, ...}
│
├── video/
│   ├── __init__.py
│   ├── handler.py          # Orchestrator — opens video, applies per-frame or temporal ops, writes output
│   ├── io.py               # OpenCV VideoCapture / VideoWriter wrappers
│   ├── temporal.py         # Frame averaging, motion blur simulation
│   ├── optical_flow.py     # Lucas-Kanade (sparse) and Farneback (dense) optical flow
│   ├── scene.py            # Scene change detection — frame differencing and histogram comparison
│   └── registry.py         # {"temporal_avg": temporal_average, "optical_flow_lk": track_points_lk, ...}
│
├── utils/
│   ├── __init__.py
│   ├── plot.py             # matplotlib helpers — waveform, spectrum, spectrogram, side-by-side image diff
│   ├── validators.py       # File existence, extension validation, MIME-based type detection
│   ├── timer.py            # @timeit decorator — logs execution time via loguru in verbose mode
│   ├── logger.py           # loguru configuration — sets level based on --verbose flag
│   └── output.py           # Timestamped output path generation — e.g. song_lowpass_20250101_120000.wav
│
├── tests/
│   ├── conftest.py         # Shared pytest fixtures — tone_440, noisy_signal, colour_image, gray_image, video_path
│   ├── fixtures/
│   │   ├── tone_440_1s.wav       # 440 Hz sine, 1 s, 44100 Hz
│   │   ├── noisy_440_1s.wav      # Same tone + Gaussian noise (SNR ~26 dB)
│   │   ├── gradient.png          # 256×256 grayscale horizontal gradient
│   │   ├── colour_blocks.png     # 256×256 four-colour BGR test image
│   │   └── test_10fps.mp4        # 30 frames, 10 fps, 64×64, alternating black/white blocks
│   ├── test_audio.py       # Tests for FFT, STFT, filters, pitch, denoise
│   ├── test_image.py       # Tests for spatial filters, 2D FFT, morphology, colour conversion
│   └── test_video.py       # Tests for temporal ops, optical flow, scene detection
│
├── scripts/
│   └── generate_fixtures.py     # Generates all test fixture files (run once, commit results)
│
├── pyproject.toml          # Project metadata, dependencies, entry point, tool configs
└── README.md               # This file
```

---

## Architecture — The Registry Pattern

Every module (`audio/`, `image/`, `video/`) uses the same architectural pattern. The registry is a `dict[str, callable]` that maps operation names to functions. The handler looks up the operation in the registry and calls it — it has no knowledge of specific operations.

### Standard operation signature (audio)

```python
def my_operation(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    # signal: shape (n_samples,) mono or (n_samples, n_channels) stereo
    # sr:     sample rate in Hz
    # params: user-supplied kwargs (cutoff, order, window, etc.)
    # returns: (modified_signal, metadata_dict)
```

### Standard operation signature (image)

```python
def my_operation(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    # img:    shape (H, W, 3) BGR uint8, or (H, W) grayscale uint8
    # params: user-supplied kwargs
    # returns: (modified_image, metadata_dict)
```

### How the handler uses the registry

```python
# audio/handler.py
from .registry import OPERATIONS
from .io import load_audio, save_audio

def handle(filepath: str, operation: str, params: dict) -> None:
    fn = OPERATIONS.get(operation)
    if not fn:
        raise ValueError(f"Unknown audio operation: {operation!r}")
    signal, sr = load_audio(filepath)
    result, meta = fn(signal, sr, params)
    save_audio(result, sr, params.get("output_path", "output/result.wav"))
```

### Adding a new operation

1. Write the function in the appropriate module file (e.g. `audio/filters.py`)
2. Add one import line and one dict entry in `audio/registry.py`

No changes to `handler.py`, `menu.py`, `cli.py`, or `main.py`. The menu auto-populates from the registry keys.

---

## Audio Module

### Fourier Analysis (FFT)

**Mathematical background:**

The Discrete Fourier Transform maps an N-point time-domain sequence x[n] to the frequency domain:

X[k] = Σ(n=0 to N-1) x[n] · e^(-j2πkn/N), k = 0, 1, ..., N-1

For real-valued signals, X[k] is conjugate-symmetric: X[N-k] = X[k]*. The single-sided spectrum retains only the first N/2 bins, with amplitudes doubled (except DC) to preserve total energy. The frequency axis in Hz is given by f[k] = k · fs / N, where fs is the sample rate.

Windowing (multiplying x[n] by a window w[n] before the DFT) reduces spectral leakage caused by the implicit rectangular window of finite-length observation. The Hann window w[n] = 0.5(1 - cos(2πn/(N-1))) is the default; it suppresses sidelobes to -31.5 dB at the cost of main-lobe width doubling relative to rectangular.

**Implementation in `audio/spectrum.py`:**

```python
def compute_fft(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    N = len(signal)
    N_fft = params.get("n_fft", N)
    window_name = params.get("window", "hann")

    from scipy.signal.windows import get_window
    w = get_window(window_name, N)
    windowed = signal * w

    X = np.fft.fft(windowed, n=N_fft)
    freqs = np.fft.fftfreq(N_fft, d=1/sr)

    half = N_fft // 2
    X_pos = np.abs(X[:half]) * 2 / N      # single-sided, amplitude-scaled
    freqs_pos = freqs[:half]

    return X_pos, {"freqs": freqs_pos, "sr": sr, "n_fft": N_fft, "window": window_name}
```

Key implementation notes:
- `np.fft.fftfreq(N_fft, d=1/sr)` returns frequencies directly in Hz (not normalised).
- The `* 2 / N` scaling corrects the single-sided amplitude so that a sinusoid at amplitude A produces a peak of A (not A/2 or AN).
- Zero-padding via `n_fft > N` interpolates the spectrum (more frequency bins) without adding real frequency resolution.

---

### Short-Time Fourier Transform (STFT)

**Mathematical background:**

The STFT analyses non-stationary signals by applying the DFT to short, overlapping segments:

X(m, k) = Σ(n=0 to L-1) x[n + mH] · w[n] · e^(-j2πkn/N)

where L is the window length (`nperseg`), H is the hop size (`nperseg - noverlap`), w[n] is the analysis window, and m is the frame index. The result is a time-frequency representation: each column of the STFT matrix shows the spectrum of one short-time frame. The inverse STFT reconstructs the signal by overlap-adding the inverse-transformed frames using the synthesis window.

The fundamental trade-off: increasing `nperseg` improves frequency resolution (Δf = fs / N) but worsens time resolution (each frame spans more time). A 75% overlap (`nperseg * 3 // 4`) is standard — it provides smooth temporal evolution while maintaining the overlap-add reconstruction conditions.

**Implementation in `audio/spectrum.py`:**

```python
from scipy.signal import stft, istft

def compute_stft(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    nperseg = params.get("nperseg", 512)
    noverlap = params.get("noverlap", nperseg * 3 // 4)
    nfft = params.get("nfft", nperseg)
    window = params.get("window", "hann")

    f, t, Zxx = stft(signal, fs=sr, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft)
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    return magnitude_db, {"freqs": f, "times": t, "sr": sr,
                          "nperseg": nperseg, "window": window}
```

The `librosa.stft()` variant is also available for audio-specific workflows (mel spectrograms, MFCCs) — it returns a complex matrix of shape `(1 + n_fft/2, n_frames)` and integrates with `librosa.display.specshow()` for plotting.

---

### IIR Filter Design and Application

**Mathematical background:**

An Nth-order IIR filter is defined by the difference equation:

y[n] = Σ(k=0 to M) b[k]·x[n-k] - Σ(k=1 to N) a[k]·y[n-k]

where b[k] and a[k] are the feedforward and feedback coefficients. For Butterworth filters, the magnitude response is maximally flat in the passband: |H(jω)|² = 1 / (1 + (ω/ωc)^(2N)), where N is the order and ωc is the cutoff. Higher orders give steeper rolloff (-20N dB/decade) but require careful numerical representation.

**Critical implementation detail — SOS vs BA:** Direct-form transfer function coefficients (`ba` format) suffer from catastrophic pole-zero sensitivity for orders above ~4–5. A 6th-order Butterworth in `ba` form can have poles move outside the unit circle in float64 arithmetic, causing instability. Second-order sections (`sos`) decompose the filter into a cascade of biquad stages, each well-conditioned. Every IIR filter in this project uses `output='sos'` with `sosfilt()` or `sosfiltfilt()`.

Zero-phase filtering (`sosfiltfilt`) runs the filter forward then backward, cancelling the phase response at the cost of non-causality (requires the entire signal in memory). The effective order doubles: an Nth-order Butterworth applied with `sosfiltfilt` has the rolloff of a 2Nth-order causal filter.

**Implementation in `audio/filters.py`:**

```python
from scipy import signal as sp_signal

def apply_lowpass(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    cutoff     = params.get("cutoff", 1000.0)
    order      = params.get("order", 4)
    zero_phase = params.get("zero_phase", False)

    sos = sp_signal.butter(order, cutoff, btype='low', fs=sr, output='sos')
    apply_fn = sp_signal.sosfiltfilt if zero_phase else sp_signal.sosfilt
    result = apply_fn(sos, signal)

    return result, {"cutoff": cutoff, "order": order, "zero_phase": zero_phase}

def apply_highpass(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    cutoff     = params.get("cutoff", 300.0)
    order      = params.get("order", 4)
    zero_phase = params.get("zero_phase", False)

    sos = sp_signal.butter(order, cutoff, btype='high', fs=sr, output='sos')
    apply_fn = sp_signal.sosfiltfilt if zero_phase else sp_signal.sosfilt
    result = apply_fn(sos, signal)

    return result, {"cutoff": cutoff, "order": order, "zero_phase": zero_phase}

def apply_bandpass(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    low  = params.get("low_cutoff", 300.0)
    high = params.get("high_cutoff", 3000.0)
    order      = params.get("order", 4)
    zero_phase = params.get("zero_phase", False)

    sos = sp_signal.butter(order, [low, high], btype='band', fs=sr, output='sos')
    apply_fn = sp_signal.sosfiltfilt if zero_phase else sp_signal.sosfilt
    result = apply_fn(sos, signal)

    return result, {"low_cutoff": low, "high_cutoff": high, "order": order, "zero_phase": zero_phase}

def apply_notch(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    freq = params.get("notch_freq", 60.0)
    Q    = params.get("Q", 30.0)

    b, a = sp_signal.iirnotch(freq / (sr / 2), Q)
    sos = sp_signal.tf2sos(b, a)
    result = sp_signal.sosfilt(sos, signal)

    return result, {"notch_freq": freq, "Q": Q}
```

Filter type selection (Butterworth, Chebyshev I/II, Elliptic) is available through a generalised designer:

```python
def design_and_apply_iir(x, sr, filter_type='butter', btype='low',
                          order=4, cutoff=1000, zero_phase=False):
    designers = {
        'butter': sp_signal.butter,
        'cheby1': lambda N, Wn, **kw: sp_signal.cheby1(N, rp=1, Wn=Wn, **kw),
        'cheby2': lambda N, Wn, **kw: sp_signal.cheby2(N, rs=40, Wn=Wn, **kw),
        'ellip':  lambda N, Wn, **kw: sp_signal.ellip(N, rp=1, rs=40, Wn=Wn, **kw),
    }
    designer = designers[filter_type]
    sos = designer(order, cutoff, btype=btype, fs=sr, output='sos')
    apply_fn = sp_signal.sosfiltfilt if zero_phase else sp_signal.sosfilt
    return apply_fn(sos, x)
```

Frequency response verification uses `sosfreqz`:

```python
from scipy.signal import sosfreqz
w, h = sosfreqz(sos, worN=8000, fs=sr)
# Plot: plt.semilogx(w, 20 * np.log10(np.abs(h) + 1e-10))
```

---

### FIR Filter Design

**Mathematical background:**

An FIR filter has no feedback (a[k] = 0 for k > 0), so it is always stable and can achieve exact linear phase when the coefficients are symmetric. The windowed-sinc method designs an FIR by computing the ideal impulse response (an infinite sinc) and truncating it with a window function. The trade-off is between filter length (computational cost) and transition bandwidth/stopband attenuation. A 101-tap Hamming-windowed FIR achieves approximately -53 dB stopband attenuation.

**Implementation in `audio/filters.py`:**

```python
from scipy.signal import firwin, fftconvolve

def apply_fir_lowpass(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    cutoff   = params.get("cutoff", 1000.0)
    numtaps  = params.get("numtaps", 101)
    window   = params.get("window", "hamming")

    taps = firwin(numtaps=numtaps, cutoff=cutoff, fs=sr, window=window, pass_zero=True)
    result = fftconvolve(signal, taps, mode='same')

    return result, {"cutoff": cutoff, "numtaps": numtaps, "window": window}
```

`fftconvolve` uses FFT-based overlap-add internally — O(N log N) vs O(NM) for direct convolution, where M is the filter length. For short filters (M < ~30), `lfilter` may be faster; for anything longer, `fftconvolve` wins.

---

### Pitch Detection

**Mathematical background:**

The autocorrelation method exploits the fact that a periodic signal with fundamental period T₀ has a peak in the autocorrelation function R(τ) at τ = T₀. The normalised autocorrelation R(τ)/R(0) reaches 1.0 at zero lag and at every integer multiple of the period. By restricting the search to lags corresponding to a musically plausible frequency range (e.g. 80–500 Hz for speech, 80–1000 Hz for music), the first significant peak gives the fundamental frequency: f₀ = fs / τ_peak.

The YIN algorithm improves on basic autocorrelation by applying a cumulative mean normalised difference function, which eliminates octave errors and provides a confidence measure. The implementation in librosa uses YIN by default.

**Implementation in `audio/pitch.py`:**

```python
import numpy as np
import librosa

def detect_pitch_autocorr(frame: np.ndarray, sr: int, params: dict) -> tuple[float, dict]:
    f_min = params.get("f_min", 80)
    f_max = params.get("f_max", 500)

    N = len(frame)
    lag_min = int(sr / f_max)
    lag_max = int(sr / f_min)

    # Autocorrelation via FFT (O(N log N) instead of O(N²))
    X = np.fft.fft(frame, n=2 * N)
    acf = np.real(np.fft.ifft(X * np.conj(X)))[:N]
    acf /= acf[0] + 1e-10   # normalise

    peak_lag = lag_min + np.argmax(acf[lag_min:lag_max])
    f0 = sr / peak_lag
    confidence = acf[peak_lag]

    return f0, {"f0": f0, "confidence": confidence, "method": "autocorr"}

def detect_pitch_yin(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    fmin = params.get("f_min", 80)
    fmax = params.get("f_max", 1000)
    frame_length = params.get("frame_length", 2048)
    hop_length = params.get("hop_length", 512)

    f0 = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=frame_length, hop_length=hop_length)
    return f0, {"f0_array": f0, "method": "yin", "sr": sr}
```

The autocorrelation method operates per-frame (the caller must frame the signal). The YIN method operates on the full signal and returns a per-frame pitch contour.

---

### Noise Reduction

**Mathematical background:**

Spectral subtraction estimates the noise power spectral density from a segment known to contain only noise (typically the first few frames), then subtracts this estimate from the noisy signal's magnitude spectrum in each frame:

|Ŝ(f)|² = max(|Y(f)|² - α · |N̂(f)|², 0)

where Y is the noisy observation, N̂ is the noise estimate, and α is an over-subtraction factor (typically 1.0–2.0) that controls aggressiveness. The half-wave rectification (max with 0) prevents negative power values, which would create musical noise artefacts. The phase of the original noisy signal is preserved — the human ear is relatively insensitive to phase, so the noisy phase is adequate for reconstruction.

**Implementation in `audio/denoise.py`:**

```python
import numpy as np
from scipy.signal import stft, istft

def spectral_subtraction(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    nperseg = params.get("nperseg", 512)
    noverlap = params.get("noverlap", 384)
    noise_frames = params.get("noise_frames", 10)
    alpha = params.get("alpha", 1.5)

    f, t, Zxx = stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    magnitude_clean = np.maximum(magnitude - alpha * noise_est, 0)

    Zxx_clean = magnitude_clean * np.exp(1j * phase)
    _, x_clean = istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)

    return x_clean[:len(signal)], {"alpha": alpha, "noise_frames": noise_frames}
```

For more advanced denoising, the `noisereduce` library provides a non-stationary Wiener-style filter:

```python
import noisereduce as nr

def wiener_denoise(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    stationary = params.get("stationary", False)
    reduced = nr.reduce_noise(y=signal, sr=sr, stationary=stationary)
    return reduced, {"method": "noisereduce", "stationary": stationary}
```

---

## Image Module

### Spatial Filtering

**Mathematical background:**

Spatial filtering applies a kernel (a small matrix) to every pixel in the image via 2D discrete convolution:

g[m, n] = Σ(i) Σ(j) h[i, j] · f[m - i, n - j]

where f is the input image and h is the kernel. For a Gaussian blur kernel, h[i,j] = (1 / 2πσ²) · exp(-(i² + j²) / 2σ²) — it weights neighbouring pixels by a Gaussian envelope, producing isotropic low-pass filtering. The Sobel operator uses two 3×3 kernels to approximate the horizontal and vertical intensity gradients:

Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

The gradient magnitude is |G| = √(Gx² + Gy²) and the edge direction is θ = atan2(Gy, Gx). Canny edge detection extends this by applying non-maximum suppression (thinning edges to 1-pixel width) and hysteresis thresholding (strong edges connected to weak edges are kept, isolated weak edges are suppressed).

**Implementation in `image/spatial.py`:**

```python
import cv2
import numpy as np

def gaussian_blur(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 5)
    sigma = params.get("sigma", 0)
    result = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return result, {"ksize": ksize, "sigma": sigma}

def apply_sobel(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result, {"ksize": ksize}

def apply_canny(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    threshold1 = params.get("threshold1", 100)
    threshold2 = params.get("threshold2", 200)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges, {"threshold1": threshold1, "threshold2": threshold2}
```

**Gotcha:** `cv2.Sobel()` outputs `CV_64F` (float64) which can be negative — you must take the absolute value or compute magnitude before converting back to uint8.

---

### 2D Frequency Domain Analysis

**Mathematical background:**

The 2D DFT extends the Fourier transform to two dimensions:

F[u, v] = Σ(m=0 to M-1) Σ(n=0 to N-1) f[m, n] · e^(-j2π(um/M + vn/N))

`np.fft.fftshift` rearranges the output so that the zero-frequency component (DC) is at the centre of the array, which is the conventional display format. The magnitude spectrum |F[u, v]| reveals the frequency content of the image — low frequencies near the centre represent smooth regions, while high frequencies at the periphery represent edges and fine detail.

Frequency-domain filtering works by multiplying the shifted spectrum by a mask H[u, v] (e.g. a circular low-pass mask that passes frequencies within a radius and blocks those outside), then inverse-transforming. An ideal low-pass filter with cutoff radius D₀ passes all frequencies with √(u² + v²) ≤ D₀ and blocks all others. However, ideal filters cause ringing artefacts (Gibbs phenomenon) — Butterworth or Gaussian filters in the frequency domain are preferred in practice.

**Implementation in `image/frequency.py`:**

```python
import numpy as np
import cv2

def image_fft(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    magnitude_db = 20 * np.log10(np.abs(F_shift) + 1)
    display = cv2.normalize(magnitude_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return display, {"shape": gray.shape}

def apply_ideal_lpf_2d(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    cutoff_fraction = params.get("cutoff_fraction", 0.1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    H, W = gray.shape

    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radius = cutoff_fraction * min(H, W) // 2
    mask = (dist <= radius).astype(float)

    F_filtered = F_shift * mask
    F_ishift = np.fft.ifftshift(F_filtered)
    result = np.real(np.fft.ifft2(F_ishift))
    return np.clip(result, 0, 255).astype(np.uint8), {"cutoff_fraction": cutoff_fraction}
```

---

### Colour Space Conversion

**Implementation in `image/colour.py`:**

OpenCV reads images as BGR by default. This module provides conversions between BGR and RGB, HSV, YCrCb, and Grayscale. HSV is useful for thresholding by hue (e.g. isolating a specific colour), while YCrCb separates luminance from chrominance — useful for operations that should operate on brightness without affecting colour.

```python
import cv2

CONVERSIONS = {
    "bgr2rgb":   cv2.COLOR_BGR2RGB,
    "bgr2hsv":   cv2.COLOR_BGR2HSV,
    "bgr2ycrcb": cv2.COLOR_BGR2YCrCb,
    "bgr2gray":  cv2.COLOR_BGR2GRAY,
    "rgb2bgr":   cv2.COLOR_RGB2BGR,
    "hsv2bgr":   cv2.COLOR_HSV2BGR,
    "ycrcb2bgr": cv2.COLOR_YCrCb2BGR,
}

def convert_colour(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    conversion = params.get("conversion", "bgr2rgb")
    code = CONVERSIONS.get(conversion)
    if code is None:
        raise ValueError(f"Unknown conversion: {conversion}")
    result = cv2.cvtColor(img, code)
    return result, {"conversion": conversion}
```

**Gotcha:** Every `cv2.imread()` returns BGR. If you pass an image to matplotlib or PIL, convert to RGB first — otherwise red and blue channels will be swapped.

---

### Morphological Operations

**Mathematical background:**

Morphological operations apply a structuring element (a binary mask, typically a rectangle or disk) to probe the image geometry. Erosion computes the minimum pixel value within the neighbourhood defined by the structuring element — it shrinks bright regions and expands dark ones. Dilation computes the maximum — it expands bright regions. Opening (erosion followed by dilation) removes small bright noise. Closing (dilation followed by erosion) fills small dark holes.

**Implementation in `image/morphology.py`:**

```python
import cv2
import numpy as np

def erode(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.erode(img, kernel)
    return result, {"ksize": ksize, "operation": "erode"}

def dilate(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.dilate(img, kernel)
    return result, {"ksize": ksize, "operation": "dilate"}

def morph_open(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return result, {"ksize": ksize, "operation": "open"}

def morph_close(img: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    ksize = params.get("ksize", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return result, {"ksize": ksize, "operation": "close"}
```

---

## Video Module

### Temporal Operations

**Implementation in `video/temporal.py`:**

Temporal operations aggregate information across frames. Frame averaging computes the mean of N consecutive frames — this reduces random noise (by a factor of √N for uncorrelated noise) and is equivalent to temporal low-pass filtering. Motion blur simulation accumulates weighted frames to simulate the effect of camera or object motion during exposure.

```python
import numpy as np
import cv2

def temporal_average(cap: cv2.VideoCapture, params: dict) -> tuple[np.ndarray, dict]:
    n_frames = params.get("n_frames", 5)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float64))
    if not frames:
        raise ValueError("No frames read")
    avg = np.mean(frames, axis=0).astype(np.uint8)
    return avg, {"n_frames_averaged": len(frames)}

def motion_blur(cap: cv2.VideoCapture, params: dict) -> tuple[np.ndarray, dict]:
    kernel_size = params.get("kernel_size", 15)
    # Create a horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read frame")
    result = cv2.filter2D(frame, -1, kernel)
    return result, {"kernel_size": kernel_size}
```

---

### Optical Flow

**Mathematical background:**

Optical flow assumes brightness constancy: I(x, y, t) ≈ I(x + dx, y + dy, t + dt). A first-order Taylor expansion gives the optical flow constraint equation:

Ix · u + Iy · v + It = 0

where (Ix, Iy, It) are the spatial and temporal image gradients, and (u, v) is the flow vector. This is one equation in two unknowns — the aperture problem. The Lucas-Kanade method resolves this by assuming the flow is constant within a local window W, yielding an overdetermined system solved via least squares:

[[ΣIx², ΣIxIy], [ΣIxIy, ΣIyIy]] · [u, v]^T = -[ΣIxIt, ΣIyIt]^T

This tracks sparse points (corners detected by `goodFeaturesToTrack`). The Farneback method fits a quadratic polynomial to each pixel's neighbourhood in both frames and derives the displacement from the polynomial coefficients, producing a dense flow field (one vector per pixel).

**Implementation in `video/optical_flow.py`:**

```python
import cv2
import numpy as np

def track_points_lk(prev_gray: np.ndarray, curr_gray: np.ndarray,
                    p0: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    lk_params = dict(
        winSize=(params.get("win_size", 15), params.get("win_size", 15)),
        maxLevel=params.get("max_level", 2),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    good_new = p1[status == 1]
    good_old = p0[status == 1]
    return good_new, {"points_tracked": len(good_new), "method": "lucas_kanade"}

def dense_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray,
                       params: dict) -> tuple[np.ndarray, dict]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=params.get("pyr_scale", 0.5),
        levels=params.get("levels", 3),
        winsize=params.get("winsize", 15),
        iterations=params.get("iterations", 3),
        poly_n=params.get("poly_n", 5),
        poly_sigma=params.get("poly_sigma", 1.2),
        flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2    # hue = direction
    hsv[..., 1] = 255                        # full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, {"method": "farneback", "mean_magnitude": float(mag.mean())}
```

**Gotcha:** `flow[y, x, 0]` is the x-displacement, `flow[y, x, 1]` is the y-displacement — the indexing is row-column, not xy.

---

### Scene Change Detection

**Mathematical background:**

Scene cuts are detected by measuring the dissimilarity between consecutive frames. The simplest metric is the mean absolute difference of pixel intensities: D = (1/MN) Σ|f_t[m,n] - f_{t-1}[m,n]|. When D exceeds a threshold, a cut is declared. For gradual transitions (fades, dissolves), histogram-based methods are more robust — the Bhattacharyya distance between the intensity histograms of consecutive frames measures distributional overlap, ranging from 0 (identical) to 1 (completely different).

**Implementation in `video/scene.py`:**

```python
import cv2
import numpy as np

def detect_scene_changes(cap: cv2.VideoCapture, params: dict) -> tuple[list[float], dict]:
    threshold = params.get("threshold", 30.0)
    method = params.get("method", "frame_diff")    # "frame_diff" or "histogram"

    ret, prev_frame = cap.read()
    if not ret:
        return [], {"cuts": []}
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    cuts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == "frame_diff":
            diff = cv2.absdiff(prev_gray, curr_gray)
            score = diff.mean()
        elif method == "histogram":
            h1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            h2 = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) * 100
        else:
            raise ValueError(f"Unknown method: {method}")

        if score > threshold:
            cuts.append(frame_idx / fps)

        prev_gray = curr_gray

    return cuts, {"cuts": cuts, "method": method, "threshold": threshold, "n_cuts": len(cuts)}
```

---

## Utility Modules

### `utils/plot.py`

matplotlib helper functions for visualising results. Provides `plot_waveform()`, `plot_spectrum()`, `plot_spectrogram()`, and `plot_image_comparison()` (side-by-side original vs processed). All functions accept numpy arrays and optional metadata, and save to the output directory.

### `utils/validators.py`

Input validation: `validate_file(filepath)` checks existence, type, and non-zero size. `detect_media_type(filepath)` returns `"audio"`, `"image"`, or `"video"` based on file extension (with MIME-type fallback via `python-magic`). Extension sets:

- Audio: `.wav`, `.flac`, `.mp3`, `.ogg`, `.aiff`, `.m4a`
- Image: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- Video: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`

### `utils/timer.py`

`@timeit` decorator that logs function execution time via loguru. Active only when verbose mode is on (DEBUG level). Uses `time.perf_counter()` for sub-millisecond resolution.

```python
def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"{fn.__module__}.{fn.__name__} took {elapsed*1000:.1f} ms")
        return result
    return wrapper
```

### `utils/logger.py`

Configures loguru with a coloured, timestamped format. `setup_logging(verbose=False)` sets INFO level; `verbose=True` sets DEBUG level. All modules import `from loguru import logger` directly — no global state beyond the initial setup call.

### `utils/output.py`

Generates timestamped output paths: `make_output_path("song.wav", "lowpass", ".wav")` → `output/song_lowpass_20250101_120000.wav`. Creates the output directory if it does not exist.

---

## Testing

Tests use pytest with shared fixtures in `tests/conftest.py`. Fixture files are generated once by `scripts/generate_fixtures.py` and committed to `tests/fixtures/`.

### Running tests

```bash
pytest tests/ -v --cov=sigforge --cov-report=term-missing
```

### Test fixture descriptions

| File | Description |
|---|---|
| `tone_440_1s.wav` | 440 Hz sine, 1 second, 44100 Hz, float32 |
| `noisy_440_1s.wav` | Same tone + Gaussian noise (SNR ~26 dB) |
| `gradient.png` | 256×256 grayscale horizontal gradient |
| `colour_blocks.png` | 256×256 four-colour BGR test image |
| `test_10fps.mp4` | 30 frames, 10 fps, 64×64, alternating black/white blocks |

### Test patterns

Each operation test follows the same structure: load fixture → apply operation → assert properties of the result and metadata. For example, `test_lowpass_attenuates_high_freq` creates a mixed signal (440 Hz + 4000 Hz), applies a 1 kHz lowpass, and verifies that the 4 kHz component is attenuated by at least 10× relative to the 440 Hz component in the frequency domain.

---

## Development Phases

| Phase | Focus | Key deliverables |
|---|---|---|
| **1 — Skeleton** | Control flow and module stubs | `main.py` → `cli.py` → `detector.py` → `menu.py` → handler stubs, registry dicts |
| **2 — Audio core** | First working operations | FFT plot, lowpass filter, spectrogram in `audio/spectrum.py` and `audio/filters.py` |
| **3 — Image core** | Spatial and frequency ops | Gaussian blur, Sobel, Canny, 2D FFT, histogram equalisation |
| **4 — Video core** | Frame-level processing | Frame extraction, apply-image-op-to-all-frames, scene detection, optical flow |
| **5 — Polish** | UX and robustness | Rich progress bars, verbose timing, full test coverage, README, CI setup |

---

## Library Stack

| Domain | Library | Key module / class | Install |
|---|---|---|---|
| Math core | NumPy | `np.fft`, `np.convolve`, `np.pad` | `pip install numpy` |
| DSP algorithms | SciPy | `scipy.signal` — filters, STFT, windows | `pip install scipy` |
| Audio I/O | soundfile | `sf.read()`, `sf.write()` | `pip install soundfile` |
| Audio format conversion | pydub | `AudioSegment.from_mp3()` | `pip install pydub` + FFmpeg |
| High-level audio analysis | librosa | `librosa.stft()`, `librosa.yin()` | `pip install librosa` |
| Image I/O + processing | OpenCV | `cv2.imread()`, `cv2.filter2D()` | `pip install opencv-python` |
| Image I/O (simple) | Pillow | `Image.open()`, `Image.save()` | `pip install Pillow` |
| Scientific image ops | scikit-image | `skimage.filters`, `skimage.morphology` | `pip install scikit-image` |
| Video I/O + frame ops | OpenCV | `cv2.VideoCapture`, `cv2.VideoWriter` | (included with opencv-python) |
| High-level video editing | moviepy | `VideoFileClip.fl_image()` | `pip install moviepy` |
| CLI args | Typer | `@app.command()`, `typer.Option()` | `pip install typer` |
| Interactive menus | InquirerPy | `inquirer.select()`, `inquirer.checkbox()` | `pip install InquirerPy` |
| Terminal output | Rich | `Console`, `Progress`, `Table` | `pip install rich` |
| Plotting | matplotlib | `plt.specgram()`, `librosa.display` | `pip install matplotlib` |
| Logging | loguru | `from loguru import logger` | `pip install loguru` |
| Noise reduction | noisereduce | `nr.reduce_noise()` | `pip install noisereduce` |
| MIME detection | python-magic | `magic.from_file()` | `pip install python-magic` |
| Type checking | mypy | `mypy sigforge/` | `pip install mypy` |
| Testing | pytest | `pytest tests/ -v --cov=sigforge` | `pip install pytest pytest-cov` |
| Formatting | black | `black sigforge/` | `pip install black` |