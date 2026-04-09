"""Cardiac cycle detection, beat extraction, and quality scoring.

Unifies approaches from:
  - MATLAB detect_cardiac_cycles.m (ABP: 50ms smoothing, 250ms distance, 10mmHg prominence)
  - Python detect_cycles() in analyze_pv_loops.py (LVP: 300ms distance, 5mmHg prominence, 20mmHg height)
  - MATLAB extract_clean_beat_segment.m (RR stability filtering, longest stable run)
  - Python _score_pv_window() (quality scoring for PV window selection)
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass
class CycleDetectionParams:
    """Configurable parameters for cardiac cycle detection."""
    smooth_window_ms: float = 50.0
    min_distance_ms: float = 250.0
    min_prominence_mmhg: float = 10.0
    min_height_mmhg: float | None = None

    @classmethod
    def abp_default(cls):
        """Conservative preset for arterial blood pressure."""
        return cls(smooth_window_ms=50, min_distance_ms=250,
                   min_prominence_mmhg=10, min_height_mmhg=None)

    @classmethod
    def lvp_default(cls):
        """Preset for left ventricular pressure (less conservative)."""
        return cls(smooth_window_ms=50, min_distance_ms=300,
                   min_prominence_mmhg=5, min_height_mmhg=20)


def _smooth_signal(signal, fs, window_ms=50.0):
    """Moving average smoothing (equivalent to MATLAB movmean)."""
    window = max(3, round(window_ms / 1000.0 * fs))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def detect_peaks(signal, fs, params=None):
    """Detect systolic peaks in a pressure waveform.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    params : CycleDetectionParams, optional
        Detection parameters. Defaults to ABP preset.

    Returns
    -------
    peak_values : np.ndarray
        Peak amplitudes (from smoothed signal).
    peak_locs : np.ndarray
        Peak sample indices (in original signal).
    """
    if params is None:
        params = CycleDetectionParams.abp_default()

    smoothed = _smooth_signal(signal, fs, params.smooth_window_ms)

    kwargs = {
        'distance': max(1, round(params.min_distance_ms / 1000.0 * fs)),
        'prominence': params.min_prominence_mmhg,
    }
    if params.min_height_mmhg is not None:
        kwargs['height'] = params.min_height_mmhg

    locs, properties = find_peaks(smoothed, **kwargs)

    if len(locs) == 0:
        return np.array([]), np.array([], dtype=int)

    peak_values = smoothed[locs]
    return peak_values, locs


def detect_cycles(signal, fs, params=None):
    """Detect cardiac cycles as peak-to-peak intervals.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    params : CycleDetectionParams, optional

    Returns
    -------
    list of (start_idx, end_idx) tuples
    """
    _, peak_locs = detect_peaks(signal, fs, params)

    if len(peak_locs) < 2:
        return []

    return [(int(peak_locs[i]), int(peak_locs[i + 1]))
            for i in range(len(peak_locs) - 1)]


def find_nadirs(signal, peak_locs):
    """Find diastolic nadirs (minima between consecutive peaks).

    Also finds the nadir before the first peak and after the last peak
    if data extends beyond.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform.
    peak_locs : np.ndarray
        Peak sample indices.

    Returns
    -------
    nadir_locs : np.ndarray
        Nadir sample indices. Length = len(peak_locs) + 1 when possible.
    """
    if len(peak_locs) == 0:
        return np.array([], dtype=int)

    nadirs = []

    # Nadir before first peak
    if peak_locs[0] > 0:
        segment = signal[:peak_locs[0]]
        nadirs.append(int(np.argmin(segment)))
    else:
        nadirs.append(0)

    # Nadirs between consecutive peaks
    for i in range(len(peak_locs) - 1):
        start = peak_locs[i]
        end = peak_locs[i + 1]
        segment = signal[start:end]
        nadirs.append(int(start + np.argmin(segment)))

    # Nadir after last peak
    if peak_locs[-1] < len(signal) - 1:
        segment = signal[peak_locs[-1]:]
        nadirs.append(int(peak_locs[-1] + np.argmin(segment)))

    return np.array(nadirs, dtype=int)


def extract_beats(signal, peak_locs, nadir_locs):
    """Extract individual beat waveforms with metadata.

    Beats are defined from nadir to nadir (diastole-to-diastole), which
    includes a complete systolic-diastolic cycle.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform.
    peak_locs : np.ndarray
        Peak sample indices.
    nadir_locs : np.ndarray
        Nadir sample indices (from find_nadirs).

    Returns
    -------
    list of dict, each with:
        'waveform': np.ndarray (the beat)
        'start': int (start sample in original signal)
        'end': int (end sample)
        'peak_idx': int (peak position relative to beat start)
        'duration_ms': float (beat duration in ms, requires fs)
    """
    beats = []
    n_peaks = len(peak_locs)

    # We need at least n_peaks + 1 nadirs for n_peaks beats
    # But we work with what we have
    for i in range(min(n_peaks, len(nadir_locs) - 1)):
        start = nadir_locs[i]
        end = nadir_locs[i + 1]

        if end <= start:
            continue

        waveform = signal[start:end].copy()
        peak_idx = int(peak_locs[i] - start)

        # Ensure peak_idx is within bounds
        if peak_idx < 0 or peak_idx >= len(waveform):
            peak_idx = int(np.argmax(waveform))

        beats.append({
            'waveform': waveform,
            'start': int(start),
            'end': int(end),
            'peak_idx': peak_idx,
        })

    return beats


def _find_consecutive_runs(logical_array, min_length):
    """Find consecutive True runs of at least min_length."""
    runs = []
    in_run = False
    run_start = 0

    for i in range(len(logical_array)):
        if logical_array[i] and not in_run:
            run_start = i
            in_run = True
        elif not logical_array[i] and in_run:
            if i - run_start >= min_length:
                runs.append((run_start, i - 1))
            in_run = False

    if in_run and len(logical_array) - run_start >= min_length:
        runs.append((run_start, len(logical_array) - 1))

    return runs


def extract_clean_segment(signal, fs, n_beats=25, rr_tolerance=0.20,
                          params=None):
    """Extract the longest stable segment of consecutive beats.

    Ports MATLAB extract_clean_beat_segment.m:
    1. Detect peaks
    2. Compute RR intervals
    3. Flag stable beats (|RR - median_RR| < rr_tolerance * median_RR)
    4. Find longest consecutive stable run >= n_beats
    5. Extract individual beat waveforms (peak-to-peak)

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    n_beats : int
        Target number of consecutive beats.
    rr_tolerance : float
        Maximum allowed RR deviation as fraction of median (default 0.20).
    params : CycleDetectionParams, optional

    Returns
    -------
    clean_beats : list of np.ndarray
        Individual beat waveforms (peak-to-peak).
    peak_locs_segment : np.ndarray
        Peak locations of the selected segment.
    info : dict
        Diagnostic info (median_rr_ms, hr_bpm, segment_indices).
    """
    _, peak_locs = detect_peaks(signal, fs, params)

    if len(peak_locs) < 3:
        return [], np.array([]), {'error': 'fewer than 3 peaks detected'}

    # RR intervals in ms
    rr_intervals = np.diff(peak_locs) / fs * 1000.0
    median_rr = np.median(rr_intervals)

    # Stability mask
    rr_stable = np.abs(rr_intervals - median_rr) < rr_tolerance * median_rr

    # Find consecutive stable runs
    runs = _find_consecutive_runs(rr_stable, min_length=n_beats)

    if runs:
        # Select longest run
        best = max(runs, key=lambda r: r[1] - r[0])
        start_beat = best[0]
        end_beat = min(best[0] + n_beats, best[1] + 1)
    else:
        # Fallback: use first n_beats
        start_beat = 0
        end_beat = min(n_beats, len(peak_locs) - 1)

    # Extract beats (peak-to-peak)
    selected_locs = peak_locs[start_beat:end_beat + 1]
    clean_beats = []
    for i in range(len(selected_locs) - 1):
        s = selected_locs[i]
        e = selected_locs[i + 1]
        clean_beats.append(signal[s:e].copy())

    return clean_beats, selected_locs, {
        'median_rr_ms': float(median_rr),
        'hr_bpm': 60000.0 / median_rr if median_rr > 0 else np.nan,
        'segment_start_peak': int(start_beat),
        'segment_end_peak': int(end_beat),
        'n_beats_extracted': len(clean_beats),
        'fallback_used': len(runs) == 0,
    }


def score_quality(signal, fs, volume=None, params=None):
    """Score signal quality of a waveform segment.

    Ports _score_pv_window() from analyze_pv_loops.py. When volume is
    provided, also scores SV magnitude and consistency.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    volume : np.ndarray, optional
        Volume waveform (for PV quality scoring).
    params : CycleDetectionParams, optional

    Returns
    -------
    dict with keys: score, n_cycles, cycle_cv, and optionally sv_raw, sv_cv
    """
    result = {'score': 0.0, 'n_cycles': 0, 'cycle_cv': np.nan}

    if len(signal) < int(fs * 2):
        return result

    _, peak_locs = detect_peaks(signal, fs, params)

    if len(peak_locs) < 3:
        return result

    cycle_lens = np.diff(peak_locs)
    med_len = np.median(cycle_lens)
    good_lens = [cl for cl in cycle_lens if abs(cl - med_len) / med_len < 0.3]

    if len(good_lens) < 2:
        return result

    cycle_cv = float(np.std(good_lens) / np.mean(good_lens))
    result['n_cycles'] = len(good_lens)
    result['cycle_cv'] = cycle_cv

    if volume is not None:
        # Smooth volume
        win = max(5, int(len(volume) / 20))
        if win % 2 == 0:
            win += 1
        win = min(win, len(volume) - 1)
        if win >= 5:
            vol_smooth = savgol_filter(volume, win, 2)
        else:
            vol_smooth = volume

        # Per-cycle stroke volume
        svs = []
        for i in range(len(peak_locs) - 1):
            s, e = peak_locs[i], peak_locs[i + 1]
            if abs((e - s) - med_len) / med_len < 0.3:
                v_cyc = vol_smooth[s:e]
                svs.append(float(v_cyc.max() - v_cyc.min()))

        if svs:
            sv_raw = float(np.median(svs))
            sv_cv = float(np.std(svs) / np.mean(svs)) if np.mean(svs) > 0 else 99.0
            result['sv_raw'] = sv_raw
            result['sv_cv'] = sv_cv

            if sv_raw >= 2:
                result['score'] = (sv_raw
                                   * (1 / (1 + sv_cv))
                                   * (1 / (1 + cycle_cv))
                                   * len(good_lens))
        else:
            result['sv_raw'] = 0.0
            result['sv_cv'] = np.nan
    else:
        # Pressure-only quality score based on cycle regularity and count
        result['score'] = len(good_lens) / (1 + cycle_cv)

    return result
