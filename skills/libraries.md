# Library Reference — SigForge Stack

## Table of contents

1. [NumPy](#numpy)
2. [SciPy signal](#scipy-signal)
3. [soundfile](#soundfile)
4. [pydub](#pydub)
5. [librosa](#librosa)
6. [OpenCV](#opencv)
7. [Pillow](#pillow)
8. [scikit-image](#scikit-image)
9. [moviepy](#moviepy)
10. [Typer](#typer)
11. [InquirerPy](#inquirerpy)
12. [Rich](#rich)
13. [loguru](#loguru)
14. [pytest](#pytest)

---

## NumPy

**Install:** `pip install numpy`  
**Import:** `import numpy as np`

Core array operations used throughout the project:

```python
# Create arrays
x = np.zeros(1024)
x = np.linspace(0, 1, 1024)         # time axis: 0 to 1 second, 1024 points

# FFT
X = np.fft.fft(x)                   # complex spectrum, length N
X_mag = np.abs(X)                    # magnitude
X_db = 20 * np.log10(X_mag + 1e-10) # magnitude in dB (avoid log(0))
freqs = np.fft.fftfreq(len(x), d=1/sr)  # frequency axis in Hz
# Keep only positive frequencies (real signals are conjugate-symmetric):
N = len(x)
X_pos = X_mag[:N//2]
freqs_pos = freqs[:N//2]

# 2D FFT for images
F = np.fft.fft2(img_gray)
F_shifted = np.fft.fftshift(F)       # zero-freq at centre
magnitude = 20 * np.log10(np.abs(F_shifted) + 1)

# Padding
x_padded = np.pad(x, (0, 512), mode='constant')  # zero-pad to next power of 2

# Dtype management
signal_float = signal.astype(np.float64)
img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
```

---

## SciPy signal

**Install:** `pip install scipy`  
**Import:** `from scipy import signal`

### Filter design

```python
from scipy import signal

# IIR filter design — ALWAYS use output='sos', never 'ba'
sos = signal.butter(N=4, Wn=1000, btype='low', fs=sr, output='sos')
sos = signal.butter(N=4, Wn=500,  btype='high', fs=sr, output='sos')
sos = signal.butter(N=4, Wn=[300, 3000], btype='band', fs=sr, output='sos')

# Apply filter (causal, introduces phase delay)
filtered = signal.sosfilt(sos, x)

# Apply filter with zero phase (forward + backward, doubles effective order)
filtered = signal.sosfiltfilt(sos, x)

# Notch filter (e.g. remove 60 Hz hum)
b, a = signal.iirnotch(60.0 / (sr / 2), Q=30)
sos = signal.tf2sos(b, a)
filtered = signal.sosfilt(sos, x)

# FIR filter design
taps = signal.firwin(numtaps=101, cutoff=1000, fs=sr, window='hamming')
filtered = signal.lfilter(taps, 1.0, x)
# or: filtered = signal.fftconvolve(x, taps, mode='same')
```

### STFT and spectrogram

```python
# STFT
f, t, Zxx = signal.stft(x, fs=sr, window='hann', nperseg=512, noverlap=256)
# f: frequency bins (Hz), t: time frames (s), Zxx: complex STFT matrix

# Magnitude spectrogram in dB
mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

# Inverse STFT
_, x_reconstructed = signal.istft(Zxx, fs=sr, nperseg=512, noverlap=256)

# Welch PSD estimate
f, Pxx = signal.welch(x, fs=sr, nperseg=1024)
```

### Window functions

```python
signal.windows.hann(512)
signal.windows.hamming(512)
signal.windows.blackman(512)
signal.windows.kaiser(512, beta=8)
```

### Resampling

```python
# Rational resampling (cleanest, no artefacts)
x_resampled = signal.resample_poly(x, up=22050, down=sr)  # resample to 22050 Hz
# or compute up/down from gcd:
from math import gcd
g = gcd(sr, target_sr)
x_resampled = signal.resample_poly(x, target_sr // g, sr // g)
```

### Convolution

```python
# For long signals, FFT-based convolution is much faster
y = signal.fftconvolve(x, h, mode='same')   # h is your impulse response
# mode options: 'full' | 'same' | 'valid'
```

---

## soundfile

**Install:** `pip install soundfile`  
**Import:** `import soundfile as sf`  
**Supports:** WAV, FLAC, OGG, AIFF (not MP3 — use pydub for that)

```python
# Read
signal, sr = sf.read("input.wav")
# signal: np.float64, range [-1.0, 1.0]
# shape: (n_samples,) for mono, (n_samples, 2) for stereo

# Write
sf.write("output.wav", signal, sr)
sf.write("output.flac", signal, sr)  # format inferred from extension

# Read metadata without loading
info = sf.info("input.wav")
print(info.samplerate, info.channels, info.duration)

# Handle stereo → mono
if signal.ndim == 2:
    mono = signal.mean(axis=1)
```

---

## pydub

**Install:** `pip install pydub` + FFmpeg must be on PATH  
**Import:** `from pydub import AudioSegment`  
**Use for:** MP3, AAC, M4A loading; simple gain/trim/concat; format conversion

```python
from pydub import AudioSegment
import numpy as np

# Load MP3
audio = AudioSegment.from_mp3("input.mp3")
audio = AudioSegment.from_file("input.m4a", format="m4a")

# Convert to numpy array for DSP
samples = np.array(audio.get_array_of_samples())
sr = audio.frame_rate
# Normalise to float [-1, 1]:
samples = samples / (2 ** (audio.sample_width * 8 - 1))

# Export
audio.export("output.mp3", format="mp3", bitrate="192k")
audio.export("output.wav", format="wav")
```

---

## librosa

**Install:** `pip install librosa`  
**Import:** `import librosa`, `import librosa.display`  
**Best for:** MFCC, CQT, onset detection, beat tracking, pitch detection, mel filterbank

```python
import librosa

# Load (returns float32, mono by default)
y, sr = librosa.load("input.wav", sr=None)  # sr=None preserves original rate
y, sr = librosa.load("input.wav", sr=22050) # resample to 22050

# STFT
D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
# D: complex (1 + n_fft/2, n_frames)

# Magnitude and phase
magnitude, phase = librosa.magphase(D)
S_db = librosa.amplitude_to_db(magnitude, ref=np.max)

# Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)

# MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Pitch estimation (YIN algorithm)
f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)

# Beat tracking
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# Harmonic-percussive separation
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Pitch shifting (semitones)
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

# Time stretching
y_stretched = librosa.effects.time_stretch(y, rate=1.5)

# Display (in matplotlib)
import librosa.display
librosa.display.waveshow(y, sr=sr)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
```

---

## OpenCV

**Install:** `pip install opencv-python`  
**Import:** `import cv2`  
**CRITICAL: OpenCV reads images as BGR. Convert to RGB before matplotlib/PIL.**

### Images

```python
import cv2
import numpy as np

# Read — returns np.uint8 BGR
img = cv2.imread("input.png")
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

# BGR → RGB (for matplotlib or PIL)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# BGR → Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# BGR → HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# BGR → YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Write (expects BGR)
cv2.imwrite("output.png", img_bgr)

# Spatial filtering
blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
blurred = cv2.blur(img, (5, 5))              # simple box filter
sharpened = cv2.filter2D(img, -1, kernel)    # custom kernel convolution

# Edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded  = cv2.erode(img, kernel)
dilated = cv2.dilate(img, kernel)
opened  = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel)
closed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Histogram equalisation
eq = cv2.equalizeHist(gray)                  # uint8 grayscale only
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq_clahe = clahe.apply(gray)
```

### Video

```python
# Read
cap = cv2.VideoCapture("input.mp4")
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame: np.uint8 BGR (height, width, 3)
    processed = my_image_op(frame)
    ...
cap.release()

# Write
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
out.write(frame_bgr)   # must be BGR uint8
out.release()

# Optical flow — Lucas-Kanade (sparse)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

# Optical flow — Farnebäck (dense)
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# flow: shape (H, W, 2) — flow[y, x] = (dx, dy)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

---

## Pillow

**Install:** `pip install Pillow`  
**Import:** `from PIL import Image`  
**Use for:** simple I/O, format conversion, thumbnail generation

```python
from PIL import Image
import numpy as np

img = Image.open("input.jpg")
img_array = np.array(img)          # RGB uint8

# Convert modes
img_gray = img.convert("L")
img_rgb  = img.convert("RGB")

# Save
img.save("output.png")
Image.fromarray(img_array).save("output.jpg", quality=95)
```

---

## scikit-image

**Install:** `pip install scikit-image`  
**Use for:** more Pythonic image ops; CLAHE; structural similarity (SSIM); Hough transform

```python
from skimage import filters, morphology, exposure, metrics
from skimage.io import imread, imsave
import numpy as np

img = imread("input.png")          # RGB uint8

# Filters
blurred  = filters.gaussian(img, sigma=2)
edges    = filters.sobel(img_gray.astype(float))
thresh   = filters.threshold_otsu(img_gray)
binary   = img_gray > thresh

# CLAHE
img_eq = exposure.equalize_adapthist(img_gray / 255.0, clip_limit=0.02)

# Morphology
from skimage.morphology import disk
eroded  = morphology.erosion(binary, disk(3))
dilated = morphology.dilation(binary, disk(3))

# Structural similarity
score = metrics.structural_similarity(img1, img2, multichannel=True)
```

---

## moviepy

**Install:** `pip install moviepy`  
**Use for:** applying an image function to every frame; exporting clips; audio extraction

```python
from moviepy.editor import VideoFileClip
import numpy as np

clip = VideoFileClip("input.mp4")

# Apply an image operation to every frame
# IMPORTANT: moviepy passes and expects RGB (not BGR)
def process_frame(frame):
    # frame: np.uint8 RGB (H, W, 3)
    # do your processing here — must return same shape/dtype
    return my_image_function(frame)

processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile("output.mp4", codec="libx264")

# Extract audio
clip.audio.write_audiofile("output.wav")

# Subclip
short = clip.subclip(t_start=10, t_end=20)  # seconds
```

---

## Typer

**Install:** `pip install typer`  
**Use for:** CLI argument parsing with type annotations

```python
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    input:     Path = typer.Argument(..., help="Path to input file"),
    operation: str  = typer.Option("fft", "--op", "-o", help="Operation to apply"),
    output:    Path = typer.Option(Path("output/"), "--output", help="Output directory"),
    verbose:   bool = typer.Option(False, "--verbose", "-v"),
):
    typer.echo(f"Processing {input} with {operation}")

if __name__ == "__main__":
    app()
```

Run: `python main.py audio.wav --op lowpass --verbose`

---

## InquirerPy

**Install:** `pip install InquirerPy`  
**Use for:** interactive terminal menus for operation selection

```python
from InquirerPy import inquirer

# Single select
operation = inquirer.select(
    message="Select operation:",
    choices=list(OPERATIONS.keys()),
).execute()

# Prompt for a parameter value
cutoff = inquirer.number(
    message="Cutoff frequency (Hz):",
    default=1000,
).execute()

# Multiple choice
ops = inquirer.checkbox(
    message="Select operations to chain:",
    choices=list(OPERATIONS.keys()),
).execute()
```

---

## Rich

**Install:** `pip install rich`  
**Use for:** pretty terminal output, progress bars, tables, logging

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# Print with markup
console.print("[bold green]✓ Done:[/bold green] output saved to output/result.wav")
console.print("[red]Error:[/red] file not found")

# Progress bar
with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as p:
    task = p.add_task("Processing frames...", total=n_frames)
    for frame in frames:
        process(frame)
        p.advance(task)

# Table (e.g. metadata summary)
table = Table(title="File Info")
table.add_column("Property")
table.add_column("Value")
table.add_row("Sample rate", str(sr))
table.add_row("Duration", f"{duration:.2f}s")
console.print(table)
```

---

## loguru

**Install:** `pip install loguru`  
**Use for:** structured logging with zero config

```python
# utils/logger.py
from loguru import logger
import sys

def setup_logging(verbose: bool = False):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

# Anywhere in the project:
from loguru import logger
logger.debug("STFT shape: {}", Zxx.shape)
logger.info("Saved output to {}", output_path)
logger.warning("Stereo input; mixing to mono")
logger.error("Unsupported format: {}", ext)
```

---

## pytest

**Install:** `pip install pytest pytest-cov`  
**Run:** `pytest tests/ -v --cov=sigforge --cov-report=term-missing`

```python
# tests/test_audio.py
import pytest
import numpy as np
from pathlib import Path
from sigforge.audio.spectrum import compute_fft

FIXTURES = Path("tests/fixtures")

@pytest.fixture
def sine_440():
    """440 Hz sine wave, 1 second, 44100 Hz sample rate."""
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t), sr

def test_fft_peak_at_440(sine_440):
    signal, sr = sine_440
    spectrum, meta = compute_fft(signal, sr, {})
    freqs = np.fft.fftfreq(len(signal), 1/sr)
    peak_freq = freqs[np.argmax(np.abs(spectrum[:len(signal)//2]))]
    assert abs(peak_freq - 440) < 1.0  # within 1 Hz

@pytest.fixture
def sample_wav():
    """Use a real small WAV from fixtures/ for integration tests."""
    return FIXTURES / "tone_1s.wav"
```
