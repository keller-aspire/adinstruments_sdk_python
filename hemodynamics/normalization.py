"""Beat phase normalization and averaging.

Ports:
  - MATLAB normalize_beats_to_cycle.m (spline to 100 points, mean/std)
  - Python average_pv_loop() from analyze_pv_loops.py (linear, filter +/-30%)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def normalize_beats(beats, n_points=100, method='spline'):
    """Phase-normalize variable-length beats to a common grid.

    Parameters
    ----------
    beats : list of np.ndarray
        Each element is a single beat waveform (variable length).
    n_points : int
        Number of output points per beat (default 100 = 1% increments).
    method : str
        Interpolation method: 'spline' (cubic, MATLAB compat) or 'linear'.

    Returns
    -------
    dict with keys:
        'normalized': np.ndarray shape (n_points, n_beats)
        'mean': np.ndarray shape (n_points,)
        'std': np.ndarray shape (n_points,)
        'cardiac_cycle_pct': np.ndarray shape (n_points,) from 0-100
    """
    if not beats:
        return None

    common_phase = np.linspace(0, 1, n_points)
    kind = 'cubic' if method == 'spline' else 'linear'

    interpolated = []
    for beat in beats:
        if len(beat) < 4:
            continue
        phase = np.linspace(0, 1, len(beat))
        try:
            f_interp = interp1d(phase, beat, kind=kind)
            interpolated.append(f_interp(common_phase))
        except Exception:
            continue

    if not interpolated:
        return None

    normalized = np.column_stack(interpolated)  # (n_points, n_beats)
    return {
        'normalized': normalized,
        'mean': np.mean(normalized, axis=1),
        'std': np.std(normalized, axis=1),
        'cardiac_cycle_pct': common_phase * 100,
    }


def average_beats(signal, fs, max_cycle_deviation=0.30, min_beats=3,
                  n_points=None, method='linear', params=None):
    """Detect cycles, filter, normalize, and average.

    Higher-level function combining cycle detection + normalization.

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform.
    fs : float
    max_cycle_deviation : float
        Filter out cycles deviating more than this fraction from median length.
    min_beats : int
        Minimum number of valid beats required.
    n_points : int, optional
        If None, uses median cycle length (matching PV pipeline behavior).
    method : str
        Interpolation method.
    params : CycleDetectionParams, optional

    Returns
    -------
    dict with 'mean', 'std', 'n_cycles', 'cycle_lengths'
    Returns None values if insufficient cycles.
    """
    from .cycles import detect_cycles as _detect_cycles

    cycles = _detect_cycles(signal, fs, params)

    if len(cycles) < min_beats:
        return {'mean': None, 'std': None, 'n_cycles': 0, 'cycle_lengths': []}

    # Filter by length
    lengths = [e - s for s, e in cycles]
    med_len = int(np.median(lengths))
    good = [(s, e) for s, e in cycles
            if abs((e - s) - med_len) / med_len < max_cycle_deviation]

    if len(good) < min_beats:
        return {'mean': None, 'std': None, 'n_cycles': 0,
                'cycle_lengths': lengths}

    # Extract beats
    beats = [signal[s:e] for s, e in good]

    # Determine n_points
    if n_points is None:
        n_points = med_len

    result = normalize_beats(beats, n_points=n_points, method=method)
    if result is None:
        return {'mean': None, 'std': None, 'n_cycles': 0,
                'cycle_lengths': lengths}

    result['n_cycles'] = len(good)
    result['cycle_lengths'] = lengths
    return result


def average_pv_loop(pressure, volume, fs, max_cycle_deviation=0.30,
                    min_beats=3, params=None):
    """Phase-normalize and average a PV loop.

    Applies volume smoothing, cycle detection on pressure, phase normalization.

    Parameters
    ----------
    pressure : np.ndarray
        LV pressure waveform (mmHg).
    volume : np.ndarray
        LV volume waveform (uncalibrated).
    fs : float
    max_cycle_deviation : float
    min_beats : int
    params : CycleDetectionParams, optional
        Defaults to LVP preset.

    Returns
    -------
    dict: 'P_mean', 'V_mean' (np.ndarray or None), 'n_cycles', 'cycle_lengths'
    """
    from .cycles import detect_cycles as _detect_cycles, CycleDetectionParams

    if params is None:
        params = CycleDetectionParams.lvp_default()

    # Smooth volume (Savitzky-Golay, window = len/20)
    win = max(5, int(len(volume) / 20))
    if win % 2 == 0:
        win += 1
    win = min(win, len(volume) - 1)
    if win >= 5:
        volume_smooth = savgol_filter(volume, win, 2)
    else:
        volume_smooth = volume.copy()

    # Detect cycles from pressure
    cycles = _detect_cycles(pressure, fs, params)

    if len(cycles) < min_beats:
        return {'P_mean': None, 'V_mean': None, 'n_cycles': 0,
                'cycle_lengths': []}

    lengths = [e - s for s, e in cycles]
    med_len = int(np.median(lengths))
    good = [(s, e) for s, e in cycles
            if abs((e - s) - med_len) / med_len < max_cycle_deviation]

    if len(good) < min_beats:
        return {'P_mean': None, 'V_mean': None, 'n_cycles': 0,
                'cycle_lengths': lengths}

    n_phase = med_len
    phase_common = np.linspace(0, 1, n_phase)

    P_interp = []
    V_interp = []

    for start, end in good:
        p_cycle = pressure[start:end]
        v_cycle = volume_smooth[start:end]

        if len(p_cycle) < 10:
            continue

        phase = np.linspace(0, 1, len(p_cycle))
        fP = interp1d(phase, p_cycle, kind='linear')
        fV = interp1d(phase, v_cycle, kind='linear')

        P_interp.append(fP(phase_common))
        V_interp.append(fV(phase_common))

    if len(P_interp) < min_beats:
        return {'P_mean': None, 'V_mean': None, 'n_cycles': 0,
                'cycle_lengths': lengths}

    return {
        'P_mean': np.mean(P_interp, axis=0),
        'V_mean': np.mean(V_interp, axis=0),
        'n_cycles': len(P_interp),
        'cycle_lengths': lengths,
    }
