# adinstruments_sdk_python

Python toolkit for reading and analyzing ADInstruments LabChart data. Two packages in one repo:

- **`adi`** -- Read `.adicht` files directly into Python/NumPy (no LabChart license needed)
- **`hemodynamics`** -- Reusable hemodynamic signal analysis built on top of `adi`

**Windows only** (ADInstruments SDK constraint).

---

## Quick Start

### Setup

```sh
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate labchart-sdk

# Option B: pip into an existing environment
git clone https://github.com/keller-aspire/adinstruments_sdk_python
cd adinstruments_sdk_python
pip install -e .
```

### Read a LabChart file

```python
import adi

f = adi.read_file('recording.adicht')
channel = f.channels[0]          # 0-indexed
data = channel.get_data(1)       # record 1 (1-indexed)

# Comments and channel metadata as DataFrames
comments = adi.extract_comments(f)
channels = adi.extract_channels(f)
```

### Analyze hemodynamics

```python
import hemodynamics as hemo

# Load from .adicht, .txt export, or preprocessed .h5
data = hemo.load_auto('recording.adicht', channel='ABP')
abp = data['signals']['ABP']
fs = data['fs']

# Continuous MAP/HR in sliding windows
df = hemo.compute_continuous_hemodynamics(abp, fs)

# Beat-by-beat feature extraction (14 features)
features = hemo.extract_all_features(abp[:60000], fs)
print(features['median'])  # maxDPDT, tau, SRT, HFER, ...

# PV loop analysis (if LVP + LVV available)
result = hemo.process_pv_loop(lvp, lvv, fs)
print(result['ESP'], result['Ea'])
```

---

## Package Structure

```
adi/                          # LabChart SDK interface
  read.py                     #   File/Channel/Record classes
  utils.py                    #   extract_comments, extract_window, etc.
  working.py                  #   EKG processing, Plotly visualization

hemodynamics/                 # Hemodynamic analysis library
  io.py                       #   Uniform loaders: .adicht, .txt, .h5
  cycles.py                   #   Cardiac cycle detection, beat extraction
  events.py                   #   Comment standardization, phase landmarks
  continuous.py               #   Sliding-window MAP/HR/SBP/DBP/PP
  normalization.py            #   Beat phase normalization, PV loop averaging
  features.py                 #   14 waveform features (dP/dt, tau, spectral)
  spectral.py                 #   FFT band power, HFER, spectral centroid
  pv_loops.py                 #   PV parameters, Ea/Ees, alpha calibration
  tests/                      #   Validation suite (pytest)
```

---

## hemodynamics Module Reference

### Data Loading (`hemodynamics.io`)

| Function | Description |
|---|---|
| `load_txt(path)` | Load LabChart .txt export (auto-detects header, columns) |
| `load_adicht(path, channels, record)` | Load .adicht via SDK with chunking for large files |
| `load_hdf5(path, animal, channels, record)` | Load preprocessed HDF5 |
| `load_auto(path)` | Auto-detect format and dispatch |

All loaders return `{'signals': {name: array}, 'fs': float, 'comments': [...], 'metadata': {...}}`.

### Cardiac Cycle Detection (`hemodynamics.cycles`)

| Function | Description |
|---|---|
| `detect_peaks(signal, fs, params)` | Find systolic peaks with configurable thresholds |
| `detect_cycles(signal, fs)` | Peak-to-peak cycle boundaries |
| `find_nadirs(signal, peak_locs)` | Diastolic minima between peaks |
| `extract_clean_segment(signal, fs, n_beats)` | Longest stable segment (RR variability filter) |
| `score_quality(signal, fs, volume)` | Signal quality scoring for window selection |

Use `CycleDetectionParams.abp_default()` or `.lvp_default()` for signal-appropriate presets.

### Continuous Hemodynamics (`hemodynamics.continuous`)

```python
df = compute_continuous_hemodynamics(abp, fs, params=ContinuousParams(window_s=60, step_s=60))
# Returns DataFrame: time_min, MAP, HR, SBP, DBP, PP
```

### Event Handling (`hemodynamics.events`)

| Function | Description |
|---|---|
| `standardize_comment(text)` | Normalize comment text to canonical labels |
| `get_phase_landmarks(comments)` | Extract protocol phase timestamps (BS, EH, ED, AR, H1-H6) |
| `extract_segment(signal, fs, center_time_s)` | Window extraction around a time point |
| `find_best_window(signal, fs, center_time_s)` | Quality-scored scanning around a landmark |

### Beat Normalization (`hemodynamics.normalization`)

| Function | Description |
|---|---|
| `normalize_beats(beats, n_points, method)` | Phase-normalize to common grid (spline or linear) |
| `average_beats(signal, fs)` | Detect, filter, normalize, and average beats |
| `average_pv_loop(pressure, volume, fs)` | PV-specific averaging with volume smoothing |

### Feature Extraction (`hemodynamics.features`)

`extract_all_features(signal, fs)` returns median and IQR across all valid beats for 14 features:

| Category | Features |
|---|---|
| Systolic | SRT (ms), SRS (mmHg/s), TmaxDPDT (ms), maxDPDT (mmHg/s) |
| Diastolic | tau (s), DiastolicR2, absMinDPDT (mmHg/s), T50 (ms) |
| Combined | DPDTR, TDE (mmHg/s), DPDTR_range (mmHg/s) |
| Spectral | HFER, SC (Hz), SSlope (dB/Hz) |

### Spectral Analysis (`hemodynamics.spectral`)

| Function | Description |
|---|---|
| `extract_spectral_features(beat, fs)` | Per-beat FFT: HFER, SC, SSlope |
| `compute_beat_averaged_spectrum(mean_beat, fs_norm)` | PSD of averaged waveform with Band 3 integration |

Default frequency bands: Band 1 (0-10 Hz), Band 2 (10-30 Hz), Band 3 (30-80 Hz). Configurable via `SpectralBands`.

### PV Loop Analysis (`hemodynamics.pv_loops`)

| Function | Description |
|---|---|
| `process_pv_loop(pressure, volume, fs)` | Full pipeline: average loop + extract parameters |
| `extract_pv_parameters(P_mean, V_mean, fs)` | ESP, EDP, dPdt_max/min, SV_raw, ESV, EDV |
| `compute_ea(esp, sv_calibrated)` | Arterial elastance |
| `compute_single_beat_ees(esp, esv_raw, v0)` | Ventricular elastance (single-beat method) |
| `compute_alpha_from_baseline(sv_raw, co, hr)` | Conductance catheter calibration factor |

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy | Array operations |
| cffi | ADInstruments DLL interface |
| scipy | Signal processing, curve fitting |
| pandas | DataFrames for tabular results |
| h5py | HDF5 file I/O |
| matplotlib | Plotting (optional) |

---

## Testing

```sh
conda activate labchart-sdk   # or your environment
python -m pytest hemodynamics/tests/ -v
```

Tests include synthetic signal validation and real-data integration tests against swine hemorrhage recordings.

---

## CFFI Build Notes (Python 3.10+)

Pre-compiled bindings are included for Python 3.6-3.9. For newer versions, compile from source:

```python
# Requires Visual Studio C++ Build Tools (v14.0+)
import os
os.chdir('adi')
exec(open("cffi_build.py").read())      # 64-bit
# exec(open("cffi_build_win32.py").read())  # 32-bit
```

---

## Origin

Fork of [Jim Hokanson's SDK](https://github.com/JimHokanson/adinstruments_sdk_python) with convenience functions by Ian Keller (`adi/utils.py`, `adi/working.py`) and hemodynamic analysis modules by Mingfeng Li (`hemodynamics/`).
