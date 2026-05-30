# Project Conventions — SigForge

Code style, type annotation patterns, testing recipes, logging, and utilities.

---

## Code style

- **PEP 8** throughout. Use `black` for formatting: `pip install black` → `black sigforge/`
- **Type annotations on all public functions.** Use `mypy sigforge/` to check.
- **Docstrings:** one-line summary + params + returns for every public function.
- **No bare `except:`.** Always catch a specific exception class.
- **Constants in ALL_CAPS** at the top of the module (e.g. `DEFAULT_SR = 22050`).

```python
# Good function signature
def apply_lowpass(
    signal: np.ndarray,
    sr: int,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """Apply a Butterworth low-pass filter to a mono audio signal.

    Args:
        signal: 1D float64 array, values in [-1.0, 1.0].
        sr:     Sample rate in Hz.
        params: Must contain 'cutoff' (Hz). Optional: 'order' (default 4),
                'zero_phase' (default False).

    Returns:
        Tuple of (filtered_signal, metadata_dict).
    """
    cutoff     = params.get("cutoff", 1000.0)
    order      = params.get("order", 4)
    zero_phase = params.get("zero_phase", False)
    ...
```

---

## Type annotations cheat sheet

```python
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Any

# Common types used in this project
AudioSignal = np.ndarray   # shape (n_samples,), dtype float64
ImageArray  = np.ndarray   # shape (H, W, 3) uint8 BGR, or (H, W) grayscale
VideoFrame  = np.ndarray   # shape (H, W, 3) uint8 BGR

OperationFn = callable[[np.ndarray, int, dict], tuple[np.ndarray, dict]]
Registry    = dict[str, OperationFn]
```

---

## The `@timeit` decorator

Implement once in `utils/timer.py` and import everywhere for verbose timing.

```python
# utils/timer.py
import time
import functools
from loguru import logger

def timeit(fn):
    """Decorator that logs execution time when verbose mode is on."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"{fn.__module__}.{fn.__name__} took {elapsed*1000:.1f} ms")
        return result
    return wrapper

# Usage:
# from utils.timer import timeit
# @timeit
# def compute_fft(signal, sr, params): ...
```

---

## Logging conventions

```python
# In every module that needs logging:
from loguru import logger

# Level guide:
logger.debug("STFT: shape={}, sr={}", Zxx.shape, sr)     # internal state, dev only
logger.info("Saved output → {}", output_path)              # user-visible progress
logger.warning("Stereo input; mixing to mono")             # non-fatal, user should know
logger.error("File not found: {}", filepath)               # recoverable error
logger.critical("Unsupported OS feature")                  # crash-level
```

Setup in `utils/logger.py`:

```python

from loguru import logger
import sys

def setup_logging(verbose: bool = False) -> None:
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", format=fmt, colorize=True)
```

---

## pytest fixtures for audio, image, video

### Generating test fixtures (run once, commit to tests/fixtures/)

```python
# scripts/generate_fixtures.py
import numpy as np
import soundfile as sf
import cv2
from pathlib import Path

OUT = Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Sine tone WAV (440 Hz, 1 s, 44100 Hz)
sr = 44100
t = np.linspace(0, 1, sr, endpoint=False)
tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
sf.write(OUT / "tone_440_1s.wav", tone, sr)

# 2. Noisy signal WAV
noisy = tone + 0.05 * np.random.randn(sr)
sf.write(OUT / "noisy_440_1s.wav", noisy.astype(np.float32), sr)

# 3. Grayscale gradient PNG (256x256)
gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
cv2.imwrite(str(OUT / "gradient.png"), gradient)

# 4. Colour test image PNG (256x256 RGB squares)
img = np.zeros((256, 256, 3), dtype=np.uint8)
img[:128, :128] = [255, 0, 0]   # blue (BGR)
img[:128, 128:] = [0, 255, 0]   # green
img[128:, :128] = [0, 0, 255]   # red
img[128:, 128:] = [128, 128, 128]
cv2.imwrite(str(OUT / "colour_blocks.png"), img)

# 5. Tiny video (30 frames, 10 fps, 64x64, alternating black/white)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUT / "test_10fps.mp4"), fourcc, 10, (64, 64))
for i in range(30):
    frame = np.full((64, 64, 3), 255 if i % 10 < 5 else 0, dtype=np.uint8)
    out.write(frame)
out.release()

print("Fixtures written to", OUT)
```

### Pytest fixture patterns

```python
# tests/conftest.py  (shared fixtures across all test files)
import pytest
import numpy as np
import soundfile as sf
import cv2
from pathlib import Path

FIXTURES = Path("tests/fixtures")

@pytest.fixture(scope="session")
def tone_440():
    """440 Hz sine, 1 second, 44100 Hz."""
    signal, sr = sf.read(FIXTURES / "tone_440_1s.wav")
    return signal, sr

@pytest.fixture(scope="session")
def noisy_signal():
    signal, sr = sf.read(FIXTURES / "noisy_440_1s.wav")
    return signal, sr

@pytest.fixture(scope="session")
def colour_image():
    """BGR uint8 test image, 256x256."""
    return cv2.imread(str(FIXTURES / "colour_blocks.png"))

@pytest.fixture(scope="session")
def gray_image():
    img = cv2.imread(str(FIXTURES / "colour_blocks.png"), cv2.IMREAD_GRAYSCALE)
    return img

@pytest.fixture
def video_path():
    return FIXTURES / "test_10fps.mp4"
```

### Writing a good operation test

```python
# tests/test_audio.py
import numpy as np
import pytest

def test_lowpass_attenuates_high_freq(tone_440):
    from sigforge.audio.filters import apply_lowpass
    signal, sr = tone_440

    # Create a mixed signal: 440 Hz (pass) + 4000 Hz (stop)
    t = np.linspace(0, 1, sr, endpoint=False)
    mixed = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 4000 * t)

    filtered, meta = apply_lowpass(mixed, sr, {"cutoff": 1000.0, "order": 4})

    # After filtering, energy in 4000 Hz band should be much lower
    fft = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), 1/sr)
    energy_4k = np.abs(fft[np.abs(freqs - 4000) < 100]).mean()
    energy_440 = np.abs(fft[np.abs(freqs - 440) < 50]).mean()

    assert energy_440 / (energy_4k + 1e-10) > 10   # 440 Hz at least 10× louder than 4 kHz

def test_fft_output_shape(tone_440):
    from sigforge.audio.spectrum import compute_fft
    signal, sr = tone_440
    result, meta = compute_fft(signal, sr, {})
    assert result.ndim == 1
    assert "freqs" in meta
```

---

## Validating input files

```python
# utils/validators.py
from pathlib import Path
import magic   # pip install python-magic

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.aiff', '.m4a'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

def validate_file(filepath: str) -> Path:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not p.is_file():
        raise ValueError(f"Not a file: {filepath}")
    if p.stat().st_size == 0:
        raise ValueError(f"File is empty: {filepath}")
    return p

def detect_media_type(filepath: str) -> str:
    """Returns 'audio', 'image', or 'video'. Raises ValueError if unsupported."""
    p = validate_file(filepath)
    ext = p.suffix.lower()

    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"

    # Fallback: MIME type from file header
    mime = magic.from_file(str(p), mime=True)
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"

    raise ValueError(f"Unsupported file type: {ext} (MIME: {mime})")
```

---

## Output directory management

```python
# utils/output.py
from pathlib import Path
import datetime

def make_output_path(input_path: str, operation: str, suffix: str,
                     output_dir: str = "output") -> Path:
    """Generate a timestamped output path.

    Example: input='song.wav', op='lowpass' → 'output/song_lowpass_20250101_120000.wav'
    """
    p = Path(input_path)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{p.stem}_{operation}_{ts}{suffix}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name
```

---

## pyproject.toml template

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "sigforge"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "soundfile>=0.12",
    "pydub>=0.25",
    "librosa>=0.10",
    "opencv-python>=4.9",
    "Pillow>=10.0",
    "scikit-image>=0.22",
    "moviepy>=1.0",
    "typer>=0.12",
    "InquirerPy>=0.3",
    "rich>=13.0",
    "loguru>=0.7",
    "python-magic>=0.4",
    "noisereduce>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.9",
    "black>=24.0",
    "ruff>=0.4",
]

[project.scripts]
sigforge = "sigforge.main:app"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=sigforge --cov-report=term-missing"
```
