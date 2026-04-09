"""Test io.py — data loading from .txt and .h5 formats."""

import os
import pytest
import numpy as np
from .conftest import N346_TXT, HDF5_PATH


class TestLoadTxt:
    """Validate .txt loading against known data properties."""

    def test_load_n346(self):
        from hemodynamics.io import load_txt
        data = load_txt(N346_TXT)

        assert 'ABP' in data['signals']
        assert 'VBP' in data['signals']
        assert data['fs'] == 1000.0
        assert len(data['signals']['ABP']) > 100_000  # multi-minute recording

        # ABP should be in physiological range
        abp = data['signals']['ABP']
        median_abp = np.nanmedian(abp)
        assert 30 < median_abp < 200, f"ABP median {median_abp} out of range"

    def test_header_detection(self):
        from hemodynamics.io import _detect_header_lines
        n = _detect_header_lines(N346_TXT)
        assert n == 9, f"Expected 9 header lines for N346, got {n}"

    def test_advanced_no_header(self):
        """Advanced .txt files have no header."""
        from hemodynamics.io import _detect_header_lines
        adv_path = os.path.join(
            r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project"
            r"\Insilico_Simulation\MATLAB_Analysis_III\Data\Advanced",
            "N781.txt")
        if os.path.exists(adv_path):
            n = _detect_header_lines(adv_path)
            assert n == 0, f"Expected 0 header lines for Advanced, got {n}"


class TestLoadHdf5:
    """Validate HDF5 loading."""

    @pytest.mark.skipif(not os.path.exists(HDF5_PATH),
                        reason="HDF5 file not available")
    def test_load_n781(self):
        from hemodynamics.io import load_hdf5
        import h5py

        with h5py.File(HDF5_PATH, 'r') as f:
            durations = f['N781'].attrs['record_durations_s']
            best_record = int(durations.argmax()) + 1

        data = load_hdf5(HDF5_PATH, 'N781', record=best_record)

        assert 'ABP' in data['signals']
        assert data['fs'] == 1000.0
        assert len(data['signals']['ABP']) > 0

    @pytest.mark.skipif(not os.path.exists(HDF5_PATH),
                        reason="HDF5 file not available")
    def test_comments_loaded(self):
        from hemodynamics.io import load_hdf5
        import h5py

        with h5py.File(HDF5_PATH, 'r') as f:
            durations = f['N781'].attrs['record_durations_s']
            best_record = int(durations.argmax()) + 1

        data = load_hdf5(HDF5_PATH, 'N781', record=best_record)
        # Should have some comments
        assert isinstance(data['comments'], list)
