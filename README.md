# adinstruments_sdk_python

Python toolkit for reading and analyzing ADInstruments LabChart data. Two packages live in this repo:

- `adi` — reads `.adicht` files directly into Python/NumPy, no LabChart license required
- `hemodynamics` — hemodynamic signal analysis built on `adi`

Windows only (ADInstruments SDK constraint). Each package can be installed independently — see [Installation](#installation).

---

## Installation

```sh
# Everything (recommended for lab use)
conda env create -f environment.yml
conda activate labchart-sdk

# Or with pip
pip install -e .            # both packages
pip install -e adi/         # reader only — just numpy + cffi
pip install -e hemodynamics/  # analysis — pulls adi-reader as a dependency
```

The reader has minimal dependencies by design. Installing `hemodynamics` separately avoids forcing scipy/pandas/h5py on users who only need file access.

---

## Usage

### Reading LabChart files

```python
import adi

f = adi.read_file('recording.adicht')
data = f.channels[0].get_data(1)   # channel 0, record 1 (1-indexed)

comments = adi.extract_comments(f)  # DataFrame
channels = adi.extract_channels(f)  # DataFrame
```

See `adi/utils.py` and `adi/working.py` for additional convenience functions (time-windowed extraction, EKG processing, Plotly visualization).

### Hemodynamic analysis

```python
import hemodynamics as hemo

# Load directly from .adicht
data = hemo.load_adicht('recording.adicht', channels=['ABP'])
abp = data['signals']['ABP']
fs = data['fs']

# Continuous hemodynamics in sliding windows
df = hemo.compute_continuous_hemodynamics(abp, fs)

# Beat-by-beat waveform features
features = hemo.extract_all_features(abp[:60000], fs)
print(features['median'])   # maxDPDT, tau, SRT, HFER, ...
print(features['n_beats'])

# PV loop analysis (requires LVP + LVV channels)
result = hemo.process_pv_loop(lvp, lvv, fs)
```

A runnable demo is included at `examples/demo_hemodynamics.py`:

```sh
python examples/demo_hemodynamics.py path/to/file.adicht
```

---

## Package overview

```
adi/                        SDK interface (Ian Keller)
  read.py                     File / Channel / Record classes
  utils.py                    Comment/channel extraction, time windows
  working.py                  EKG processing, Plotly visualization

hemodynamics/               Signal analysis (Mingfeng Li)
  io.py                       Data loaders (.adicht, .txt, .h5)
  cycles.py                   Peak detection, beat extraction, quality scoring
  events.py                   Comment standardization, protocol phase landmarks
  continuous.py               Sliding-window MAP / HR / SBP / DBP / PP
  normalization.py            Beat phase normalization, PV loop averaging
  features.py                 14 waveform features per beat (see below)
  spectral.py                 FFT band power, spectral centroid, spectral slope
  pv_loops.py                 PV parameters, Ea / Ees, conductance calibration

examples/                   Usage demos
scripts/                    Dev-only validation (not part of the package)
```

### Extracted waveform features

`extract_all_features()` computes median and IQR across all valid beats in a segment:

- **Systolic:** SRT (ms), SRS (mmHg/s), TmaxDPDT (ms), maxDPDT (mmHg/s)
- **Diastolic:** tau (s), DiastolicR2, absMinDPDT (mmHg/s), T50 (ms)
- **Combined:** DPDTR, TDE (mmHg/s), DPDTR_range (mmHg/s)
- **Spectral:** HFER, SC (Hz), SSlope (dB/Hz)

Spectral features use three frequency bands by default: 0–10 Hz (cardiac fundamental), 10–30 Hz (harmonics), 30–80 Hz (high-frequency). Configurable via `SpectralBands`.

### Peak detection presets

`CycleDetectionParams` ships with two presets:
- `.abp_default()` — prominence 10 mmHg, min distance 250 ms, no height filter
- `.lvp_default()` — prominence 5 mmHg, min distance 300 ms, height 20 mmHg

All thresholds are overridable.

---

## Dependencies

`adi` requires only **numpy** and **cffi**.

`hemodynamics` additionally requires **scipy**, **pandas**, and **h5py**. Matplotlib is optional (plotting only).

---

## Testing

```sh
python -m pytest hemodynamics/tests/ -v
```

The test suite includes synthetic signal tests (always runnable) and integration tests against lab recordings (skipped if data files are not present).

---

## CFFI build notes

Pre-compiled bindings are included for Python 3.6–3.13 (64-bit). If you need to recompile:

```python
# Requires Visual Studio C++ Build Tools (v14.0+)
import os
os.chdir('adi')
exec(open("cffi_build.py").read())
```

---

## Contributors

Original SDK by [Jim Hokanson](https://github.com/JimHokanson/adinstruments_sdk_python). Convenience functions and repo maintenance by Ian Keller. Hemodynamic analysis modules by Mingfeng Li.
