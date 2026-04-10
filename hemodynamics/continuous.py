"""Sliding-window continuous hemodynamics computation.

Ports MATLAB compute_continuous_hemodynamics.m:
  - Sliding windows with configurable duration and stride
  - Per-window: peak detection -> HR, SBP, DBP, PP, MAP
  - Physiological range validation
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .cycles import detect_peaks, CycleDetectionParams


@dataclass
class ContinuousParams:
    """Parameters for continuous hemodynamics computation."""
    window_s: float = 60.0
    step_s: float = 60.0
    hr_range_bpm: tuple = (20.0, 300.0)
    map_range_mmhg: tuple = (5.0, 250.0)
    signal_range_mmhg: tuple = (10.0, 300.0)
    min_peaks_per_window: int = 3
    min_valid_seconds: float = 5.0


def _compute_segment_hemodynamics(segment, fs, params=None):
    """Compute hemodynamic metrics from a single ABP segment.

    Ports MATLAB compute_segment_hemodynamics local function.

    Returns
    -------
    dict with MAP, HR, SBP, DBP, PP (NaN for failures).
    """
    result = {k: np.nan for k in ('MAP', 'HR', 'SBP', 'DBP', 'PP')}

    if params is None:
        params = ContinuousParams()

    # Remove NaN
    clean = segment[~np.isnan(segment)]
    if len(clean) < fs * params.min_valid_seconds:
        return result

    # Quality check
    sig_range = float(clean.max() - clean.min())
    if sig_range < params.signal_range_mmhg[0] or sig_range > params.signal_range_mmhg[1]:
        return result

    # Detect peaks — use default ABP params (prominence + distance only).
    # No height threshold: prominence is sufficient and avoids silent
    # failure on signals with different baselines or unit scaling.
    peak_values, peak_locs = detect_peaks(clean, fs)

    if len(peak_locs) < params.min_peaks_per_window:
        return result

    # HR from inter-beat intervals
    ibi = np.diff(peak_locs) / fs  # seconds
    valid_ibi = ibi[(ibi > 0.24) & (ibi < 2.0)]  # 30-250 bpm range
    if len(valid_ibi) < 2:
        return result

    hr = 60.0 / float(np.mean(valid_ibi))

    # Diastolic values: minimum between consecutive peaks
    diastolic_vals = []
    for i in range(len(peak_locs) - 1):
        cycle_seg = clean[peak_locs[i]:peak_locs[i + 1]]
        if len(cycle_seg) > 0:
            diastolic_vals.append(float(np.min(cycle_seg)))

    if not diastolic_vals:
        return result

    sbp = float(np.median(peak_values))
    dbp = float(np.median(diastolic_vals))
    pp = sbp - dbp
    map_val = dbp + pp / 3.0

    # Validate ranges
    if hr < params.hr_range_bpm[0] or hr > params.hr_range_bpm[1]:
        hr = np.nan
    if map_val < params.map_range_mmhg[0] or map_val > params.map_range_mmhg[1]:
        map_val = sbp = dbp = pp = np.nan

    result.update({
        'MAP': map_val, 'HR': hr, 'SBP': sbp, 'DBP': dbp, 'PP': pp,
    })
    return result


def compute_continuous_hemodynamics(signal, fs, params=None):
    """Compute MAP, HR, SBP, DBP, PP in sliding windows.

    Parameters
    ----------
    signal : np.ndarray
        ABP waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    params : ContinuousParams, optional

    Returns
    -------
    pd.DataFrame with columns: time_min, MAP, HR, SBP, DBP, PP
    """
    if params is None:
        params = ContinuousParams()

    window_samples = int(params.window_s * fs)
    step_samples = int(params.step_s * fs)
    n_samples = len(signal)

    n_windows = max(0, (n_samples - window_samples) // step_samples + 1)

    if n_windows < 1:
        return pd.DataFrame(columns=['time_min', 'MAP', 'HR', 'SBP', 'DBP', 'PP'])

    rows = []
    for w in range(n_windows):
        start_idx = w * step_samples
        end_idx = start_idx + window_samples
        center_sample = (start_idx + end_idx) / 2
        time_min = center_sample / fs / 60.0

        segment = signal[start_idx:end_idx]
        metrics = _compute_segment_hemodynamics(segment, fs, params)
        metrics['time_min'] = time_min
        rows.append(metrics)

    return pd.DataFrame(rows)[['time_min', 'MAP', 'HR', 'SBP', 'DBP', 'PP']]
