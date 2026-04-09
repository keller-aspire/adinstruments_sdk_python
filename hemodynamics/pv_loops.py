"""Pressure-volume loop analysis.

Ports from analyze_pv_loops.py:
  - extract_pv_parameters(): ESP, EDP, dPdt, SV_raw, ESV, EDV
  - Ea/Ees computation and alpha calibration
"""

import numpy as np


def extract_pv_parameters(P_mean, V_mean, fs=1000.0):
    """Extract hemodynamic parameters from an averaged PV loop.

    Parameters
    ----------
    P_mean : np.ndarray
        Averaged pressure loop (mmHg).
    V_mean : np.ndarray
        Averaged volume loop (uncalibrated).
    fs : float
        Effective sampling frequency of the averaged loop.

    Returns
    -------
    dict with:
        ESP, EDP, dev_pressure (mmHg)
        dPdt_max, dPdt_min (mmHg/s)
        SV_raw, ESV_raw, EDV_raw (uncalibrated)
    """
    if P_mean is None or V_mean is None:
        return {}

    # Pressure-derived (always reliable)
    ESP = float(np.max(P_mean))
    EDP = float(np.min(P_mean))
    dev_pressure = ESP - EDP

    dt = 1.0 / fs
    dPdt = np.gradient(P_mean) / dt
    dPdt_max = float(np.max(dPdt))
    dPdt_min = float(np.min(dPdt))

    # Volume-derived (uncalibrated conductance catheter)
    ed_idx = int(np.argmax(V_mean))
    es_idx = int(np.argmin(V_mean))
    EDV_raw = float(V_mean[ed_idx])
    ESV_raw = float(V_mean[es_idx])
    SV_raw = EDV_raw - ESV_raw

    return {
        'ESP': ESP,
        'EDP': EDP,
        'dev_pressure': dev_pressure,
        'dPdt_max': dPdt_max,
        'dPdt_min': dPdt_min,
        'SV_raw': SV_raw,
        'ESV_raw': ESV_raw,
        'EDV_raw': EDV_raw,
    }


def compute_alpha_from_baseline(sv_raw_baseline, co_baseline, hr_baseline):
    """Compute conductance catheter calibration factor.

    alpha = SV_true_baseline / SV_raw_baseline
    where SV_true = CO / HR * 1000

    Parameters
    ----------
    sv_raw_baseline : float
        Uncalibrated stroke volume at baseline (mL).
    co_baseline : float
        Cardiac output at baseline (L/min).
    hr_baseline : float
        Heart rate at baseline (bpm).

    Returns
    -------
    float: alpha calibration factor
    """
    if hr_baseline <= 0 or sv_raw_baseline <= 0 or co_baseline <= 0:
        return np.nan
    sv_true = co_baseline / hr_baseline * 1000.0  # mL
    return sv_true / sv_raw_baseline


def calibrate_sv(sv_raw, alpha):
    """Apply volume calibration: SV_cal = alpha * SV_raw."""
    if np.isnan(alpha) or alpha <= 0:
        return np.nan
    return sv_raw * alpha


def compute_ea(esp, sv_calibrated):
    """Compute effective arterial elastance: Ea = ESP / SV_cal (mmHg/mL)."""
    if sv_calibrated <= 0 or np.isnan(sv_calibrated):
        return np.nan
    return esp / sv_calibrated


def compute_single_beat_ees(esp, esv_raw, v0):
    """Compute single-beat end-systolic elastance.

    Ees = ESP / (ESV_raw - V0) (mmHg/mL)

    Parameters
    ----------
    esp : float
        End-systolic pressure (mmHg).
    esv_raw : float
        Raw end-systolic volume (uncalibrated).
    v0 : float
        Volume-axis intercept of ESPVR (from IVC occlusion baseline).

    Returns
    -------
    float: Ees (mmHg/mL), or NaN if physiologically implausible
    """
    denom = esv_raw - v0
    if denom <= 0:
        return np.nan
    ees = esp / denom
    # Sanity check: physiological range
    if ees <= 0 or ees > 10:
        return np.nan
    return float(ees)


def process_pv_loop(pressure, volume, fs):
    """Full PV loop pipeline: average loop -> extract parameters.

    Parameters
    ----------
    pressure : np.ndarray
        LV pressure waveform (mmHg).
    volume : np.ndarray
        LV volume waveform (uncalibrated).
    fs : float

    Returns
    -------
    dict with all PV parameters plus n_cycles, cycle_lengths, HR_pv
    """
    from .normalization import average_pv_loop

    result = average_pv_loop(pressure, volume, fs)

    if result['P_mean'] is None:
        return {'n_cycles': 0, 'cycle_lengths': result.get('cycle_lengths', [])}

    params = extract_pv_parameters(result['P_mean'], result['V_mean'], fs)

    # HR from median cycle length
    if result['cycle_lengths']:
        med_len = np.median(result['cycle_lengths'])
        params['HR_pv'] = 60.0 / (med_len / fs)
    else:
        params['HR_pv'] = np.nan

    params['n_cycles'] = result['n_cycles']
    params['cycle_lengths'] = result['cycle_lengths']

    return params
