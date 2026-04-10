# Update Notes

## 2026-04-11

**Fixed CFFI path resolution in sub-package setup files.** `adi/setup.py` and `hemodynamics/setup.py` now use `os.path.abspath` to resolve `package_dir` correctly regardless of whether `pip install` is invoked from the repo root or the sub-directory. Previously, `package_dir={"adi": "."}` caused CFFI modules (`.pyd` files) not to be found when installing via `pip install -e adi/`.

**Fixed signal range check rejecting windows with transient artifacts.** `_compute_segment_hemodynamics()` was using raw `min`/`max` to compute signal range, so a single spike (e.g., from a flush or zeroing event) would push the range above the 300 mmHg threshold and discard the entire window. Now uses 1st–99th percentile range instead. Tested on a 20-channel ALI recording with artifact-heavy records — all records now produce valid output.

---

## 2026-04-10

**Repo reorganization.** Moved dev-only validation scripts to `scripts/`. Added `examples/demo_hemodynamics.py` (CLI demo that accepts any `.adicht` or `.txt` file). Updated `.gitignore` to cover build artifacts (`*.egg-info`, `__pycache__`, `.pytest_cache`). Cleaned up stale egg-info directories.

**Split into two installable packages.** `adi/setup.py` installs the reader alone (numpy + cffi). `hemodynamics/setup.py` installs the analysis package and declares adi-reader as a dependency. The root `setup.py` remains as a convenience wrapper for installing both at once. This keeps the reader lightweight for users who don't need scipy/pandas/h5py.

**Fixed peak detection in `continuous.py`.** `compute_continuous_hemodynamics()` was using an adaptive height threshold (`np.percentile(signal, 60)`) that caused silent failure on signals with different baselines or scaling — `detect_peaks` would return empty arrays. Removed the height threshold; prominence (10 mmHg) and minimum distance (250 ms) are sufficient. Height filtering remains available through `CycleDetectionParams` for specialized use (e.g., LVP). Cross-validation unchanged (SRT: 166 ms vs MATLAB 167 ms; DBP: 75.4 vs 75.39 mmHg). All 25 tests pass.

## 2026-04-09

**Initial `hemodynamics` package.** Eight modules: `io.py` (data loading from .adicht, .txt, .h5), `cycles.py` (cardiac cycle detection, beat extraction), `events.py` (LabChart comment standardization, protocol phase landmarks), `continuous.py` (sliding-window MAP/HR/SBP/DBP/PP), `normalization.py` (beat phase normalization, PV loop averaging), `features.py` (14 per-beat waveform features), `spectral.py` (FFT band analysis), `pv_loops.py` (PV parameters, Ea/Ees, conductance calibration). Added `environment.yml`, test suite, and updated README.
