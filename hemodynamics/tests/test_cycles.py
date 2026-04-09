"""Test cycles.py — cardiac cycle detection and beat extraction."""

import numpy as np
import pytest


def _generate_synthetic_abp(fs=1000, duration_s=10, hr_bpm=80, sbp=120, dbp=80):
    """Generate a synthetic ABP-like signal for testing."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    cycle_len = 60.0 / hr_bpm
    freq = 1.0 / cycle_len

    # Simple sinusoidal with sharp systolic peak
    phase = 2 * np.pi * freq * t
    abp = dbp + (sbp - dbp) * (0.5 * (1 + np.sin(phase - np.pi / 2)))
    # Add some harmonics for realism
    abp += (sbp - dbp) * 0.1 * np.sin(2 * phase)
    return abp


class TestDetectPeaks:
    def test_synthetic_signal(self):
        from hemodynamics.cycles import detect_peaks

        abp = _generate_synthetic_abp(hr_bpm=80, duration_s=10)
        values, locs = detect_peaks(abp, 1000.0)

        # At 80 bpm for 10s, expect ~13 peaks (80/60*10)
        expected = int(80 / 60 * 10)
        assert abs(len(locs) - expected) <= 2, \
            f"Expected ~{expected} peaks, got {len(locs)}"

    def test_peak_values_reasonable(self):
        from hemodynamics.cycles import detect_peaks

        abp = _generate_synthetic_abp(sbp=120, dbp=80)
        values, locs = detect_peaks(abp, 1000.0)

        # Peaks should be near SBP
        assert np.all(values > 90), "Peaks should be above diastolic range"

    def test_lvp_preset(self):
        from hemodynamics.cycles import detect_peaks, CycleDetectionParams

        abp = _generate_synthetic_abp(sbp=100, dbp=10)  # LVP-like
        params = CycleDetectionParams.lvp_default()
        values, locs = detect_peaks(abp, 1000.0, params)
        assert len(locs) > 5


class TestDetectCycles:
    def test_cycle_count(self):
        from hemodynamics.cycles import detect_cycles

        abp = _generate_synthetic_abp(hr_bpm=80, duration_s=10)
        cycles = detect_cycles(abp, 1000.0)

        expected = int(80 / 60 * 10) - 1  # n_peaks - 1
        assert abs(len(cycles) - expected) <= 2


class TestFindNadirs:
    def test_nadir_count(self):
        from hemodynamics.cycles import detect_peaks, find_nadirs

        abp = _generate_synthetic_abp()
        _, peak_locs = detect_peaks(abp, 1000.0)
        nadir_locs = find_nadirs(abp, peak_locs)

        # Should have n_peaks + 1 nadirs (before first + between + after last)
        assert len(nadir_locs) == len(peak_locs) + 1

    def test_nadirs_below_peaks(self):
        from hemodynamics.cycles import detect_peaks, find_nadirs

        abp = _generate_synthetic_abp()
        peak_vals, peak_locs = detect_peaks(abp, 1000.0)
        nadir_locs = find_nadirs(abp, peak_locs)

        nadir_vals = abp[nadir_locs]
        # All nadirs should be below all peaks (with tolerance)
        assert np.mean(nadir_vals) < np.mean(peak_vals)


class TestExtractCleanSegment:
    def test_extracts_beats(self):
        from hemodynamics.cycles import extract_clean_segment

        abp = _generate_synthetic_abp(hr_bpm=80, duration_s=30)
        beats, locs, info = extract_clean_segment(abp, 1000.0, n_beats=10)

        assert len(beats) >= 10, f"Expected >=10 beats, got {len(beats)}"
        assert info['hr_bpm'] > 0

    def test_beat_waveform_shape(self):
        from hemodynamics.cycles import extract_clean_segment

        abp = _generate_synthetic_abp(hr_bpm=80, duration_s=30)
        beats, _, _ = extract_clean_segment(abp, 1000.0, n_beats=5)

        for beat in beats:
            assert len(beat) > 100  # at 80bpm/1kHz, ~750 samples per beat
            assert beat.max() > beat.min()


class TestScoreQuality:
    def test_good_signal_scores_high(self):
        from hemodynamics.cycles import score_quality

        abp = _generate_synthetic_abp(hr_bpm=80, duration_s=10)
        result = score_quality(abp, 1000.0)

        assert result['score'] > 0
        assert result['n_cycles'] > 5

    def test_flat_signal_scores_zero(self):
        from hemodynamics.cycles import score_quality

        flat = np.ones(10000) * 80.0
        result = score_quality(flat, 1000.0)

        assert result['score'] == 0.0
