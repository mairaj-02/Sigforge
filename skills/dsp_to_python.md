# DSP Concepts → Python Code — SigForge Reference

This file maps signals & systems theory (as you'd know it from an EE curriculum) to
concrete Python implementations in the SigForge stack. Assumes NumPy, SciPy, librosa,
and OpenCV are available.

## Table of contents
1. [Fourier analysis](#fourier-analysis)
2. [Filter design workflow](#filter-design-workflow)
3. [STFT and time-frequency analysis](#stft-and-time-frequency-analysis)
4. [Sampling and resampling](#sampling-and-resampling)
5. [Convolution and correlation](#convolution-and-correlation)
6. [2D image frequency domain](#2d-image-frequency-domain)
7. [Noise reduction](#noise-reduction)
8. [Pitch detection](#pitch-detection)
9. [Optical flow (video)](#optical-flow-video)
10. [Scene change detection](#scene-change-detection)

---

## Fourier analysis

### Continuous → Discrete mapping

| Theory | Python |
|---|---|
| X(jω) — CTFT | Not computable directly; approximate with very high Fs |
| X[k] — DFT, N points | `np.fft.fft(x, n=N)` |
| Frequency bins | `np.fft.fftfreq(N, d=1/sr)` → Hz axis |
| Positive frequencies only | `X[:N//2]`, `freqs[:N//2]` |
| Magnitude spectrum | `np.abs(X)` |
| Phase spectrum | `np.angle(X)` |
| Power spectrum | `np.abs(X)**2 / N` |
| dB magnitude | `20 * np.log10(np.abs(X) + 1e-10)` |
| Parseval's theorem check | `np.sum(x**2)` ≈ `np.sum(np.abs(X)**2) / N` |

### Full example: FFT of a real signal

```python
import numpy as np

def compute_fft(signal: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    N = len(signal)
    # Optionally zero-pad to next power of 2 for speed
    N_fft = params.get("n_fft", N)

    # Apply window to reduce spectral leakage
    window_name = params.get("window", "hann")
    window = getattr(np.lib.stride_tricks, "as_strided", None)
    from scipy.signal.windows import get_window
    w = get_window(window_name, N)
    windowed = signal * w

    X = np.fft.fft(windowed, n=N_fft)
    freqs = np.fft.fftfreq(N_fft, d=1/sr)

    # Return single-sided (positive freqs only)
    half = N_fft // 2
    X_pos = np.abs(X[:half]) * 2 / N      # scale so amplitude is correct
    freqs_pos = freqs[:half]

    return X_pos, {"freqs": freqs_pos, "sr": sr, "n_fft": N_fft, "window": window_name}
```

**Why windowing?** Spectral leakage from the rectangular window (finite observation)
smears energy into neighbouring bins. Hann window is the default safe choice. Use
Blackman for better sidelobe suppression; Kaiser for tunable rolloff.

---

## Filter design workflow

### The right workflow (IIR, causal)

```
1. Specify: filter type, order N, cutoff(s) in Hz, sample rate fs
2. Design: scipy.signal.butter / cheby1 / cheby2 / ellip → SOS form
3. Apply: scipy.signal.sosfilt (causal) or sosfiltfilt (zero-phase)
4. Verify: plot frequency response with freqz_sos
```

```python
from scipy import signal
import numpy as np

def design_and_apply_iir(x, sr, filter_type='butter', btype='low',
                          order=4, cutoff=1000, zero_phase=False):
    # Normalised cutoff: Wn = f_cutoff / (sr/2)
    # When you pass fs=sr, scipy handles the normalisation for you
    designers = {
        'butter': signal.butter,
        'cheby1': lambda N, Wn, **kw: signal.cheby1(N, rp=1, Wn=Wn, **kw),
        'cheby2': lambda N, Wn, **kw: signal.cheby2(N, rs=40, Wn=Wn, **kw),
        'ellip':  lambda N, Wn, **kw: signal.ellip(N, rp=1, rs=40, Wn=Wn, **kw),
    }
    designer = designers[filter_type]
    sos = designer(order, cutoff, btype=btype, fs=sr, output='sos')

    apply_fn = signal.sosfiltfilt if zero_phase else signal.sosfilt
    return apply_fn(sos, x)
```

### Verifying frequency response

```python
from scipy.signal import sosfreqz
import matplotlib.pyplot as plt

sos = signal.butter(4, 1000, 'low', fs=sr, output='sos')
w, h = sosfreqz(sos, worN=8000, fs=sr)
plt.semilogx(w, 20 * np.log10(np.abs(h) + 1e-10))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True)
plt.title("Butterworth LPF — frequency response")
plt.show()
```

### FIR filter design

```python
from scipy.signal import firwin, lfilter, fftconvolve

# Parks-McClellan / windowed sinc
taps = firwin(numtaps=101,          # must be odd for Type I linear phase
              cutoff=1000,
              fs=sr,
              window='hamming',
              pass_zero=True)        # True = LPF, False = HPF

filtered = fftconvolve(x, taps, mode='same')   # FFT convolution for long signals
# or: filtered = lfilter(taps, 1.0, x)         # direct form (slower for long)
```

### Filter type comparison

| Filter | Passband | Stopband | Phase | Use when |
|---|---|---|---|---|
| Butterworth | Maximally flat | Monotone rolloff | Non-linear | General purpose, no ripple OK |
| Chebyshev I | Equiripple | Monotone | Non-linear | Sharper rolloff, passband ripple OK |
| Chebyshev II | Flat | Equiripple | Non-linear | Stopband attenuation is critical |
| Elliptic | Equiripple | Equiripple | Non-linear | Sharpest rolloff, both ripples OK |
| FIR (windowed)| Varies | Varies | **Linear** | Phase response matters (audio) |

---

## STFT and time-frequency analysis

### SciPy STFT

```python
from scipy.signal import stft, istft
import numpy as np

# Forward STFT
f, t, Zxx = stft(x, fs=sr,
                 window='hann',
                 nperseg=512,       # analysis window length (samples)
                 noverlap=256,      # overlap = nperseg - hop_length
                 nfft=512)          # FFT size (≥ nperseg; pad if larger)
# f: (nperseg//2 + 1,) frequency bins in Hz
# t: (n_frames,) time frames in seconds
# Zxx: complex (nperseg//2 + 1, n_frames)

magnitude = np.abs(Zxx)
phase     = np.angle(Zxx)
magnitude_db = 20 * np.log10(magnitude + 1e-10)

# Inverse STFT (reconstruction)
_, x_hat = istft(Zxx, fs=sr, window='hann', nperseg=512, noverlap=256)
```

### Parameter tuning guide

| Parameter | Effect | Typical values |
|---|---|---|
| `nperseg` | Time-frequency resolution trade-off | 256, 512, 1024, 2048 |
| `noverlap` | Temporal resolution; 75% overlap common | `nperseg * 3 // 4` |
| `nfft` | Frequency resolution (zero-pads window) | Power of 2 ≥ nperseg |
| Window | Sidelobe leakage vs main lobe width | hann (default), blackman |

**Resolution trade-off (Heisenberg-Gabor limit):**
- Large `nperseg` → better frequency resolution, worse time resolution
- Small `nperseg` → better time resolution, worse frequency resolution
- Rule of thumb for speech: `nperseg = 25ms * sr` (e.g. 512 at 22050 Hz)

### librosa STFT (preferred for audio analysis)

```python
import librosa

D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048, window='hann')
# D: complex (1 + n_fft/2, n_frames) = (1025, n_frames) for n_fft=2048

S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Mel spectrogram (perceptual scale)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)
```

---

## Sampling and resampling

```python
from scipy.signal import resample_poly
from math import gcd

def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down)
    # Internally: FIR anti-aliasing filter + polyphase decomposition

# Nyquist: to represent a signal with bandwidth B Hz, you need sr > 2B
# If sr_out < sr_in, frequencies above sr_out/2 will alias — resample_poly handles this
```

---

## Convolution and correlation

```python
from scipy.signal import fftconvolve, correlate
import numpy as np

# Linear convolution (direct output length = len(x) + len(h) - 1)
y_full = fftconvolve(x, h, mode='full')

# Same length output (centred; what you usually want for filtering)
y_same = fftconvolve(x, h, mode='same')

# Cross-correlation (find delay between two signals)
corr = correlate(x, ref, mode='full')
lags = np.arange(-(len(ref)-1), len(x))   # lag axis in samples
delay_samples = lags[np.argmax(corr)]
delay_seconds = delay_samples / sr
```

---

## 2D image frequency domain

```python
import numpy as np
import cv2

def image_fft(img_bgr: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)          # zero frequency at centre

    magnitude = np.abs(F_shift)
    magnitude_db = 20 * np.log10(magnitude + 1)  # +1 to avoid log(0)

    # Normalise to uint8 for display
    display = cv2.normalize(magnitude_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return display, {"shape": gray.shape}

def apply_ideal_lpf_2d(img_bgr, cutoff_fraction=0.1):
    """cutoff_fraction: fraction of image half-width (0 = DC only, 1 = full image)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape

    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)

    # Build circular mask
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radius = cutoff_fraction * min(H, W) // 2
    mask = (dist <= radius).astype(float)

    F_filtered = F_shift * mask
    F_ishift   = np.fft.ifftshift(F_filtered)
    result     = np.real(np.fft.ifft2(F_ishift))
    return np.clip(result, 0, 255).astype(np.uint8)
```

---

## Noise reduction

### Spectral subtraction (simple, fast)

```python
import numpy as np
from scipy.signal import stft, istft

def spectral_subtraction(x: np.ndarray, sr: int, params: dict):
    nperseg = params.get("nperseg", 512)
    noverlap = params.get("noverlap", 384)
    noise_frames = params.get("noise_frames", 10)   # first N frames assumed noise

    f, t, Zxx = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase     = np.angle(Zxx)

    # Estimate noise PSD from first noise_frames frames
    noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Subtract noise estimate (half-wave rectify to prevent negative values)
    alpha = params.get("alpha", 1.5)   # over-subtraction factor (1–2)
    magnitude_clean = np.maximum(magnitude - alpha * noise_est, 0)

    # Reconstruct
    Zxx_clean = magnitude_clean * np.exp(1j * phase)
    _, x_clean = istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=noverlap)

    return x_clean[:len(x)], {"noise_est_db": 20*np.log10(noise_est.mean() + 1e-10)}
```

### Wiener filter approach (SciPy)

```python
# scipy doesn't have a direct audio Wiener filter, but does have:
from scipy.signal import wiener

# Wiener filter for images (not audio)
img_denoised = wiener(img_noisy, mysize=5)   # 5x5 local neighbourhood

# For audio, use spectral subtraction or noisereduce library:
# pip install noisereduce
import noisereduce as nr
reduced = nr.reduce_noise(y=x, sr=sr, stationary=False)
```

---

## Pitch detection

### Autocorrelation method

```python
import numpy as np

def detect_pitch_autocorr(frame: np.ndarray, sr: int, f_min=80, f_max=500) -> float:
    """Returns fundamental frequency in Hz, or 0 if unvoiced."""
    N = len(frame)
    # Lag range corresponding to f_min and f_max
    lag_min = int(sr / f_max)
    lag_max = int(sr / f_min)

    # Autocorrelation via FFT (fast)
    X = np.fft.fft(frame, n=2*N)
    acf = np.real(np.fft.ifft(X * np.conj(X)))[:N]
    acf /= acf[0] + 1e-10   # normalise

    # Find peak in valid lag range
    peak_lag = lag_min + np.argmax(acf[lag_min:lag_max])
    f0 = sr / peak_lag
    confidence = acf[peak_lag]   # 1 = perfectly periodic, 0 = noise
    return f0 if confidence > 0.5 else 0.0
```

### YIN (librosa — preferred)

```python
import librosa

f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=512)
# f0: array of per-frame pitch estimates (Hz); 0 where unvoiced
```

---

## Optical flow (video)

### Lucas-Kanade (sparse — tracks specific points)

```python
import cv2
import numpy as np

def track_points_lk(cap: cv2.VideoCapture, max_corners=200):
    ret, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect corners to track
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.3,
                          minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

        good_new = p1[status == 1]
        good_old = p0[status == 1]
        # good_new[i] is where good_old[i] moved to

        prev_gray = curr_gray
        p0 = good_new.reshape(-1, 1, 2)
```

### Farnebäck (dense — flow at every pixel)

```python
def dense_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5,   # pyramid scale (0.5 = classic)
        levels=3,        # pyramid levels
        winsize=15,      # averaging window size
        iterations=3,    # per pyramid level
        poly_n=5,        # neighbourhood for poly expansion
        poly_sigma=1.2,  # Gaussian std for poly weights
        flags=0
    )
    # flow[y, x, 0] = x-displacement, flow[y, x, 1] = y-displacement

    # Visualise as HSV (hue=direction, value=magnitude)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ang * 180 / np.pi / 2   # hue = direction
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow, bgr
```

---

## Scene change detection

```python
import cv2
import numpy as np

def detect_scene_changes(cap: cv2.VideoCapture, threshold=30.0) -> list[float]:
    """Returns list of timestamps (seconds) where scene cuts occur."""
    ret, prev_frame = cap.read()
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

        # Mean absolute difference between consecutive frames
        diff = cv2.absdiff(prev_gray, curr_gray)
        score = diff.mean()

        if score > threshold:
            timestamp = frame_idx / fps
            cuts.append(timestamp)

        prev_gray = curr_gray

    return cuts
```

For more robust detection, use histogram comparison:
```python
def histogram_diff(f1_gray, f2_gray) -> float:
    h1 = cv2.calcHist([f1_gray], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([f2_gray], [0], None, [256], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    # Returns 0 (identical) to 1 (completely different)
```
