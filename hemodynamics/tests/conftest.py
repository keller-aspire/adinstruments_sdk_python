"""Shared test fixtures and paths for hemodynamics validation tests."""

import os
import pytest

# === Base Paths ===
PROJECT_BASE = r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project"

# SwineHemorrhage (MATLAB analysis with .txt data)
SWINE_HEM_DIR = os.path.join(PROJECT_BASE, "SwineHemorrhage")
DATA_TXT_DIR = os.path.join(SWINE_HEM_DIR, "Data_txt_1000Hz")
RESULTS_DIR = os.path.join(SWINE_HEM_DIR, "Results_Refined_Analysis")

# Microcirculation (Python analysis with .adicht / .h5 data)
MICRO_DIR = os.path.join(PROJECT_BASE, "SwineHemorrhage_Microcirculation")
LABCHART_DIR = os.path.join(MICRO_DIR, "LabChart")
HDF5_PATH = os.path.join(LABCHART_DIR, "hemodynamics.h5")
CACHE_DIR = os.path.join(MICRO_DIR, "Microcirculation_Analysis", ".cache")

# Insilico (MATLAB analysis with .txt data)
INSILICO_DIR = os.path.join(PROJECT_BASE, "Insilico_Simulation",
                            "MATLAB_Analysis_III")
INSILICO_DATA_DIR = os.path.join(INSILICO_DIR, "Data")

# === Validation Animal Paths ===
# Primary ABP validation (Group 1, good quality)
N346_TXT = os.path.join(DATA_TXT_DIR, "Group1", "N346.txt")
# Secondary ABP validation (Group 2)
N376_TXT = os.path.join(DATA_TXT_DIR, "Group2", "N376.txt")
# Advanced animal (all signals) — .adicht + .h5
N781_ADICHT = os.path.join(LABCHART_DIR, "N781.adicht")
N782_ADICHT = os.path.join(LABCHART_DIR, "N782.adicht")

# === Tolerance Constants ===
TOL_MAP_MMHG = 2.0        # MAP tolerance
TOL_HR_BPM = 2.0          # HR tolerance
TOL_DPDT_FRAC = 0.05      # 5% for dP/dt features
TOL_TAU_FRAC = 0.10       # 10% for tau (solver-sensitive)
TOL_SPECTRAL_FRAC = 0.05  # 5% for spectral features
TOL_ESP_MMHG = 1.0        # ESP tolerance
TOL_BEAT_CORR = 0.99      # Beat shape correlation minimum


# === Fixtures ===
@pytest.fixture
def n346_data():
    """Load N346 .txt data."""
    from hemodynamics.io import load_txt
    return load_txt(N346_TXT)


@pytest.fixture
def n781_hdf5_data():
    """Load N781 from HDF5 (record with longest duration)."""
    from hemodynamics.io import load_hdf5
    import h5py
    with h5py.File(HDF5_PATH, 'r') as f:
        durations = f['N781'].attrs['record_durations_s']
        best_record = int(durations.argmax()) + 1
    return load_hdf5(HDF5_PATH, 'N781', record=best_record)
