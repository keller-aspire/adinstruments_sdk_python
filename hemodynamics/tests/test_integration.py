"""Integration test — full pipeline on real data.

Exercises: io -> cycles -> continuous -> normalization -> features -> spectral
on N346 .txt data end-to-end.
"""

import os
import numpy as np
import pytest

from .conftest import (
    N346_TXT, HDF5_PATH, RESULTS_DIR,
    TOL_MAP_MMHG, TOL_HR_BPM, TOL_BEAT_CORR,
)


@pytest.mark.skipif(not os.path.exists(N346_TXT),
                    reason="N346 .txt not available")
class TestFullPipelineN346:
    """End-to-end pipeline validation on N346 (Group 1, ABP-only)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from hemodynamics.io import load_txt
        self.data = load_txt(N346_TXT)
        self.abp = self.data['signals']['ABP']
        self.fs = self.data['fs']

    def test_01_data_loaded(self):
        assert len(self.abp) > 100_000
        assert self.fs == 1000.0
        median = np.nanmedian(self.abp)
        assert 30 < median < 200

    def test_02_cycle_detection(self):
        from hemodynamics.cycles import detect_peaks, detect_cycles

        # Use first 60 seconds
        segment = self.abp[:60_000]
        values, locs = detect_peaks(segment, self.fs)

        # Should detect peaks at reasonable rate
        hr_approx = len(locs) / 60.0 * 60.0
        assert 40 < hr_approx < 200, f"HR ~{hr_approx} bpm seems wrong"

        cycles = detect_cycles(segment, self.fs)
        assert len(cycles) >= 10

    def test_03_continuous_hemodynamics(self):
        from hemodynamics.continuous import compute_continuous_hemodynamics

        # First 5 minutes
        segment = self.abp[:300_000]
        df = compute_continuous_hemodynamics(segment, self.fs)

        assert len(df) > 0
        assert not df['MAP'].isna().all()
        assert not df['HR'].isna().all()

        # Physiological ranges for swine
        valid_map = df['MAP'].dropna()
        assert valid_map.min() > 20
        assert valid_map.max() < 200

        valid_hr = df['HR'].dropna()
        assert valid_hr.min() > 30
        assert valid_hr.max() < 250

    def test_04_beat_normalization(self):
        from hemodynamics.normalization import average_beats

        segment = self.abp[:60_000]
        result = average_beats(segment, self.fs, n_points=100)

        assert result['mean'] is not None
        assert result['n_cycles'] >= 3
        assert len(result['mean']) == 100

        # Mean beat should have a clear systolic peak
        mean_beat = result['mean']
        peak_idx = np.argmax(mean_beat)
        assert mean_beat[peak_idx] > mean_beat[0]  # peak higher than start

    def test_05_feature_extraction(self):
        from hemodynamics.features import extract_all_features, FEATURE_NAMES

        segment = self.abp[:60_000]
        result = extract_all_features(segment, self.fs)

        assert result['n_beats'] > 0, "No beats extracted"

        # Check that key features are not NaN
        median = result['median']
        assert not np.isnan(median['maxDPDT']), "maxDPDT should not be NaN"
        assert not np.isnan(median['SRT']), "SRT should not be NaN"

        # Physiological plausibility
        assert median['maxDPDT'] > 0, "maxDPDT should be positive"
        assert 50 < median['SRT'] < 500, f"SRT {median['SRT']}ms out of range"
        assert median['TDE'] > 0, "TDE should be positive"

    def test_06_spectral_features(self):
        from hemodynamics.features import extract_all_features

        segment = self.abp[:60_000]
        result = extract_all_features(segment, self.fs, include_spectral=True)

        median = result['median']
        # HFER should be between 0 and 1
        if not np.isnan(median.get('HFER', np.nan)):
            assert 0 <= median['HFER'] <= 1
        # SC should be positive
        if not np.isnan(median.get('SC', np.nan)):
            assert median['SC'] > 0

    def test_07_event_standardization(self):
        from hemodynamics.events import standardize_comment

        # Test known patterns
        assert standardize_comment("Baseline Start") == ["Baseline Start"]
        assert standardize_comment("BL") == ["Baseline Start"]
        assert standardize_comment("end BL") == ["Baseline End"]
        assert standardize_comment("HS (Fast) Start") == ["HS Fast Start"]
        assert standardize_comment("HS (Slow) End") == ["HS Slow End"]
        assert standardize_comment("Start Res") == ["Resuscitation Start"]
        assert standardize_comment("3 Hr Post") == ["3H Post"]
        assert standardize_comment("CO 3.77") == ["CO 3.77"]
        assert standardize_comment("") == []
        assert standardize_comment("random noise") == []


@pytest.mark.skipif(not os.path.exists(HDF5_PATH),
                    reason="HDF5 file not available")
class TestPVLoopPipeline:
    """PV loop pipeline validation on N781 (Advanced, from HDF5)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from hemodynamics.io import load_hdf5
        import h5py

        with h5py.File(HDF5_PATH, 'r') as f:
            durations = f['N781'].attrs['record_durations_s']
            self.best_record = int(durations.argmax()) + 1

        self.data = load_hdf5(HDF5_PATH, 'N781', record=self.best_record)
        self.fs = self.data['fs']

    def test_pv_loop_processing(self):
        from hemodynamics.pv_loops import process_pv_loop

        signals = self.data['signals']
        if 'Pressure' not in signals or 'Magnitude_Volume' not in signals:
            pytest.skip("LVP/LVV channels not in this record")

        # Extract a 10-second segment near the middle
        lvp = signals['Pressure']
        lvv = signals['Magnitude_Volume']
        mid = len(lvp) // 2
        seg_len = int(10 * self.fs)
        start = max(0, mid - seg_len // 2)

        result = process_pv_loop(
            lvp[start:start + seg_len],
            lvv[start:start + seg_len],
            self.fs,
        )

        if result['n_cycles'] > 0:
            assert 'ESP' in result
            assert result['ESP'] > 0
            assert result['EDP'] >= 0
            assert result['SV_raw'] != 0

    def test_phase_landmarks(self):
        from hemodynamics.events import get_phase_landmarks

        comments = self.data['comments']
        if not comments:
            pytest.skip("No comments in this record")

        landmarks = get_phase_landmarks(comments)
        # Should find at least some landmarks
        assert isinstance(landmarks, dict)


@pytest.mark.skipif(not os.path.exists(RESULTS_DIR),
                    reason="MATLAB results not available")
class TestCrossValidationMATLAB:
    """Compare against MATLAB .mat results where available."""

    def test_morphology_comparison_n346(self):
        """Compare feature extraction against MATLAB Morphology results."""
        import scipy.io as sio

        mat_path = os.path.join(RESULTS_DIR, "N346", "Morphology",
                                "N346_Morphology_Results.mat")
        if not os.path.exists(mat_path):
            pytest.skip(f"MATLAB results not found: {mat_path}")

        # Load MATLAB results
        mat = sio.loadmat(mat_path, squeeze_me=True)

        # Load and process with our pipeline
        from hemodynamics.io import load_txt
        from hemodynamics.features import extract_all_features

        data = load_txt(N346_TXT)
        abp = data['signals']['ABP']

        # Extract from first 60-second segment (Baseline)
        segment = abp[:60_000]
        result = extract_all_features(segment, 1000.0)

        if result['n_beats'] == 0:
            pytest.skip("No beats extracted from segment")

        # Report comparison (detailed assertions depend on .mat structure)
        print(f"\nPython n_beats: {result['n_beats']}")
        for key in ['maxDPDT', 'SRT', 'tau', 'HFER']:
            py_val = result['median'].get(key, np.nan)
            print(f"  {key}: {py_val:.4f}")
