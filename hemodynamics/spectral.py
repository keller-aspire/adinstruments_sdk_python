"""Spectral analysis of pressure waveforms.

Ports:
  - MATLAB extract_spectral_features.m (per-beat FFT)
  - MATLAB compute_beat_averaged_spectrum.m (beat-averaged PSD)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SpectralBands:
    """Frequency band definitions for ABP spectral analysis."""
    band1: tuple = (0, 10)      # Cardiac fundamental
    band2: tuple = (10, 30)     # Cardiac harmonics
    band3: tuple = (30, 80)     # High-frequency


def extract_spectral_features(beat, fs, nfft=1024, bands=None):
    """Extract spectral features from a single beat.

    Parameters
    ----------
    beat : np.ndarray
        Single beat waveform (mmHg).
    fs : float
        Sampling frequency (Hz).
    nfft : int
        FFT length (zero-padded).
    bands : SpectralBands, optional

    Returns
    -------
    dict with HFER (dimensionless), SC (Hz), SSlope (dB/Hz)
    """
    if bands is None:
        bands = SpectralBands()

    result = {'HFER': np.nan, 'SC': np.nan, 'SSlope': np.nan}

    try:
        n = len(beat)
        if n < 4:
            return result

        # Detrend and window
        signal = beat - np.mean(beat)
        window = np.hanning(n)
        windowed = signal * window

        # FFT
        fft_result = np.fft.rfft(windowed, n=nfft)

        # PSD normalized by fs and window power
        window_power = np.sum(window ** 2)
        psd = np.abs(fft_result) ** 2 / (fs * window_power)
        freq = np.fft.rfftfreq(nfft, d=1.0 / fs)

        # Band power integration (trapezoidal)
        def band_power(f_low, f_high):
            mask = (freq >= f_low) & (freq <= f_high)
            if np.sum(mask) < 2:
                return 0.0
            return float(np.trapz(psd[mask], freq[mask]))

        p1 = band_power(*bands.band1)
        p2 = band_power(*bands.band2)
        p3 = band_power(*bands.band3)
        total = p1 + p2 + p3

        # HFER
        if total > 0:
            result['HFER'] = float(p3 / total)

        # Spectral Centroid (>= 1 Hz)
        sc_mask = freq >= 1.0
        if np.sum(sc_mask) > 0 and np.sum(psd[sc_mask]) > 0:
            result['SC'] = float(
                np.sum(freq[sc_mask] * psd[sc_mask]) / np.sum(psd[sc_mask]))

        # Spectral Slope (10-80 Hz, dB/Hz)
        slope_mask = (freq >= 10) & (freq <= 80)
        if np.sum(slope_mask) > 2:
            freq_s = freq[slope_mask]
            psd_db = 10.0 * np.log10(psd[slope_mask] + np.finfo(float).eps)
            coeffs = np.polyfit(freq_s, psd_db, 1)
            result['SSlope'] = float(coeffs[0])

    except Exception:
        pass

    return result


def compute_beat_averaged_spectrum(mean_beat, fs_normalized,
                                  freq_range=(0, 100), nfft=None):
    """Compute PSD of a beat-averaged waveform.

    Parameters
    ----------
    mean_beat : np.ndarray
        Beat-averaged waveform (from normalization.normalize_beats).
    fs_normalized : float
        Equivalent sampling frequency after normalization.
        e.g. 100 points at 80 bpm -> 100 / (60/80) = 133.3 Hz
    freq_range : tuple
        (f_min, f_max) in Hz.
    nfft : int, optional
        FFT length. Defaults to max(1024, next power of 2).

    Returns
    -------
    dict with:
        'freq': np.ndarray (Hz)
        'psd_db': np.ndarray (dB)
        'band3_power_db': float (integrated Band 3 power in dB)
    """
    n = len(mean_beat)

    if nfft is None:
        nfft = max(1024, int(2 ** np.ceil(np.log2(n))))

    # Detrend and window
    signal = mean_beat - np.mean(mean_beat)
    window = np.hanning(n)
    windowed = signal * window

    # FFT
    fft_result = np.fft.rfft(windowed, n=nfft)

    # PSD
    window_power = np.sum(window ** 2)
    psd = np.abs(fft_result) ** 2 / (fs_normalized * window_power)
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs_normalized)

    # Convert to dB
    psd_db = 10.0 * np.log10(psd + np.finfo(float).eps)

    # Restrict to range
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq_out = freq[mask]
    psd_db_out = psd_db[mask]

    # Band 3 power (30-80 Hz)
    b3_mask = (freq_out >= 30) & (freq_out <= 80)
    if np.sum(b3_mask) >= 2:
        psd_linear = 10.0 ** (psd_db_out[b3_mask] / 10.0)
        b3_power = float(np.trapz(psd_linear, freq_out[b3_mask]))
        b3_power_db = float(10.0 * np.log10(b3_power + np.finfo(float).eps))
    else:
        b3_power_db = np.nan

    return {
        'freq': freq_out,
        'psd_db': psd_db_out,
        'band3_power_db': b3_power_db,
    }
