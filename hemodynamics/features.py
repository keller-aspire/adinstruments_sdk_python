"""Waveform feature extraction from arterial blood pressure beats.

Ports MATLAB extract_*_features.m suite:
  - 4 systolic features: SRT, SRS, TmaxDPDT, maxDPDT
  - 4 diastolic features: tau, DiastolicR2, absMinDPDT, T50
  - 3 combined features: DPDTR, TDE, DPDTR_range
  - 3 spectral features: HFER, SC, SSlope (delegated to spectral.py)

All per-beat extractions are wrapped in try/except; failures return NaN.
Aggregation uses median + IQR (robust to outliers).
"""

import numpy as np
from scipy.optimize import curve_fit

FEATURE_NAMES = [
    'SRT', 'SRS', 'TmaxDPDT', 'maxDPDT',
    'tau', 'DiastolicR2', 'absMinDPDT', 'T50',
    'DPDTR', 'TDE', 'DPDTR_range',
    'HFER', 'SC', 'SSlope',
]


def extract_systolic_features(beat, fs, peak_idx, nadir_idx=0):
    """Extract systolic features from a single beat.

    Parameters
    ----------
    beat : np.ndarray
        Single beat waveform (mmHg), nadir-to-nadir.
    fs : float
        Sampling frequency (Hz).
    peak_idx : int
        Index of systolic peak within beat.
    nadir_idx : int
        Index of diastolic nadir (default 0 = start of beat).

    Returns
    -------
    dict with SRT (ms), SRS (mmHg/s), TmaxDPDT (ms), maxDPDT (mmHg/s)
    """
    result = {k: np.nan for k in ('SRT', 'SRS', 'TmaxDPDT', 'maxDPDT')}
    try:
        dt = 1.0 / fs
        dpdt = np.gradient(beat) / dt

        # Systolic Rise Time
        srt_ms = (peak_idx - nadir_idx) / fs * 1000.0
        result['SRT'] = srt_ms

        # Systolic Rise Slope
        sbp = beat[peak_idx]
        dbp = beat[nadir_idx]
        pp = sbp - dbp
        if srt_ms > 0:
            result['SRS'] = pp / (srt_ms / 1000.0)

        # Max dP/dt during systolic upstroke
        upstroke = dpdt[nadir_idx:peak_idx + 1]
        if len(upstroke) > 0:
            max_idx_rel = int(np.argmax(upstroke))
            result['maxDPDT'] = float(upstroke[max_idx_rel])
            result['TmaxDPDT'] = max_idx_rel / fs * 1000.0
    except Exception:
        pass
    return result


def _exp_decay(t, A, tau, B):
    """Exponential decay model: P(t) = A * exp(-t/tau) + B."""
    return A * np.exp(-t / tau) + B


def extract_diastolic_features(beat, fs, peak_idx, next_nadir_idx=None):
    """Extract diastolic features from a single beat.

    Parameters
    ----------
    beat : np.ndarray
        Single beat waveform (mmHg).
    fs : float
    peak_idx : int
        Index of systolic peak within beat.
    next_nadir_idx : int, optional
        End of beat. Defaults to len(beat) - 1.

    Returns
    -------
    dict with tau (s), DiastolicR2, absMinDPDT (mmHg/s), T50 (ms)
    """
    if next_nadir_idx is None:
        next_nadir_idx = len(beat) - 1

    result = {k: np.nan for k in ('tau', 'DiastolicR2', 'absMinDPDT', 'T50')}
    try:
        dt = 1.0 / fs
        dpdt = np.gradient(beat) / dt

        # Absolute min dP/dt in diastolic phase
        diastolic_dpdt = dpdt[peak_idx:next_nadir_idx + 1]
        if len(diastolic_dpdt) > 0:
            result['absMinDPDT'] = float(np.abs(np.min(diastolic_dpdt)))

        # Exponential decay fitting
        fit_start = peak_idx + round(0.05 * fs)  # 50ms after peak
        fit_end = next_nadir_idx

        if fit_start < fit_end and (fit_end - fit_start) >= 10:
            diastolic_p = beat[fit_start:fit_end + 1].astype(np.float64)
            t_fit = np.arange(len(diastolic_p)) / fs

            P0 = diastolic_p[0]
            Pend = diastolic_p[-1]
            A_init = max(P0 - Pend, 1.0)
            B_init = max(Pend, 0.0)

            try:
                popt, _ = curve_fit(
                    _exp_decay, t_fit, diastolic_p,
                    p0=[A_init, 0.3, B_init],
                    bounds=([0, 0.05, 0], [200, 2.0, 200]),
                    maxfev=5000,
                )
                tau = popt[1]
                fitted = _exp_decay(t_fit, *popt)
                ss_res = np.sum((diastolic_p - fitted) ** 2)
                ss_tot = np.sum((diastolic_p - np.mean(diastolic_p)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                result['tau'] = float(tau)
                result['DiastolicR2'] = float(r2)
            except Exception:
                pass

        # Half-decay time (T50)
        sbp = beat[peak_idx]
        dbp = beat[next_nadir_idx]
        p_half = (sbp + dbp) / 2.0

        diastolic_seg = beat[peak_idx:next_nadir_idx + 1]
        crossings = np.where(diastolic_seg <= p_half)[0]
        if len(crossings) > 0 and crossings[0] > 0:
            idx = crossings[0]
            P1 = diastolic_seg[idx - 1]
            P2 = diastolic_seg[idx]
            if P1 != P2:
                frac = (p_half - P1) / (P2 - P1)
            else:
                frac = 0.0
            result['T50'] = (idx - 1 + frac) / fs * 1000.0

    except Exception:
        pass
    return result


def extract_combined_features(beat, fs, max_dpdt, abs_min_dpdt):
    """Extract combined sharpness features.

    Parameters
    ----------
    beat : np.ndarray
        Single beat waveform.
    fs : float
    max_dpdt : float
        From systolic extraction.
    abs_min_dpdt : float
        From diastolic extraction.

    Returns
    -------
    dict with DPDTR, TDE (mmHg/s), DPDTR_range (mmHg/s)
    """
    result = {k: np.nan for k in ('DPDTR', 'TDE', 'DPDTR_range')}
    try:
        dt = 1.0 / fs
        dpdt = np.gradient(beat) / dt

        if abs_min_dpdt > 0 and not np.isnan(max_dpdt):
            result['DPDTR'] = float(max_dpdt / abs_min_dpdt)

        result['TDE'] = float(np.mean(np.abs(dpdt)))

        if not np.isnan(max_dpdt) and not np.isnan(abs_min_dpdt):
            result['DPDTR_range'] = float(max_dpdt + abs_min_dpdt)
    except Exception:
        pass
    return result


def extract_beat_features(beat, fs, peak_idx, nadir_idx=0,
                          next_nadir_idx=None, include_spectral=True):
    """Extract all features from a single beat.

    Returns dict with all 14 (or 11 without spectral) feature keys.
    """
    if next_nadir_idx is None:
        next_nadir_idx = len(beat) - 1

    features = {}

    # Systolic
    sys_f = extract_systolic_features(beat, fs, peak_idx, nadir_idx)
    features.update(sys_f)

    # Diastolic
    dia_f = extract_diastolic_features(beat, fs, peak_idx, next_nadir_idx)
    features.update(dia_f)

    # Combined
    comb_f = extract_combined_features(
        beat, fs, sys_f.get('maxDPDT', np.nan), dia_f.get('absMinDPDT', np.nan))
    features.update(comb_f)

    # Spectral
    if include_spectral:
        try:
            from .spectral import extract_spectral_features
            spec_f = extract_spectral_features(beat, fs)
            features.update(spec_f)
        except ImportError:
            features.update({k: np.nan for k in ('HFER', 'SC', 'SSlope')})

    return features


def extract_all_features(signal, fs, beat_duration_range_ms=(250, 2000),
                         include_spectral=True, params=None):
    """Extract features from an ABP segment across all valid beats.

    Parameters
    ----------
    signal : np.ndarray
        ABP waveform (mmHg).
    fs : float
    beat_duration_range_ms : tuple
        (min, max) beat duration in ms. Beats outside are skipped.
    include_spectral : bool
        Include spectral features (HFER, SC, SSlope).
    params : CycleDetectionParams, optional

    Returns
    -------
    dict with keys:
        'median': dict of feature_name -> median value
        'iqr': dict of feature_name -> IQR value
        'n_beats': int
        'per_beat': list of dicts (one per beat)
    """
    from .cycles import detect_peaks, find_nadirs

    _, peak_locs = detect_peaks(signal, fs, params)

    if len(peak_locs) < 2:
        nan_dict = {k: np.nan for k in FEATURE_NAMES}
        return {'median': nan_dict, 'iqr': nan_dict, 'n_beats': 0, 'per_beat': []}

    nadir_locs = find_nadirs(signal, peak_locs)

    # Number of complete beats = min(n_peaks, n_nadirs - 1)
    n_cycles = min(len(peak_locs), len(nadir_locs) - 1)

    per_beat = []
    min_dur, max_dur = beat_duration_range_ms

    for i in range(n_cycles):
        beat_start = nadir_locs[i]
        beat_end = nadir_locs[i + 1]
        peak_abs = peak_locs[i]

        duration_ms = (beat_end - beat_start) / fs * 1000.0
        if duration_ms < min_dur or duration_ms > max_dur:
            continue

        beat = signal[beat_start:beat_end + 1].copy()
        peak_rel = int(peak_abs - beat_start)

        # Bounds check
        if peak_rel < 0 or peak_rel >= len(beat):
            peak_rel = int(np.argmax(beat))

        try:
            features = extract_beat_features(
                beat, fs,
                peak_idx=peak_rel,
                nadir_idx=0,
                next_nadir_idx=len(beat) - 1,
                include_spectral=include_spectral,
            )
            per_beat.append(features)
        except Exception:
            continue

    if not per_beat:
        nan_dict = {k: np.nan for k in FEATURE_NAMES}
        return {'median': nan_dict, 'iqr': nan_dict, 'n_beats': 0, 'per_beat': []}

    # Aggregate: median + IQR
    feature_keys = FEATURE_NAMES if include_spectral else FEATURE_NAMES[:11]
    median_dict = {}
    iqr_dict = {}

    for key in feature_keys:
        values = np.array([b.get(key, np.nan) for b in per_beat], dtype=np.float64)
        valid = values[~np.isnan(values)]
        if len(valid) > 0:
            median_dict[key] = float(np.median(valid))
            iqr_dict[key] = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        else:
            median_dict[key] = np.nan
            iqr_dict[key] = np.nan

    return {
        'median': median_dict,
        'iqr': iqr_dict,
        'n_beats': len(per_beat),
        'per_beat': per_beat,
    }
