"""Hemodynamics analysis library for ADInstruments LabChart data.

Built on top of the adi SDK. Provides reusable signal processing modules
for cardiac cycle detection, hemodynamic parameter extraction, waveform
feature analysis, and PV loop analysis.
"""

__version__ = "0.1.0"

from .io import load_txt, load_adicht, load_hdf5, load_auto
from .cycles import (
    detect_peaks, detect_cycles, find_nadirs,
    extract_beats, extract_clean_segment, score_quality,
    CycleDetectionParams,
)
from .continuous import compute_continuous_hemodynamics, ContinuousParams
from .events import (
    standardize_comment, get_phase_landmarks,
    extract_segment, find_best_window,
)
from .normalization import normalize_beats, average_beats, average_pv_loop
from .features import extract_all_features, FEATURE_NAMES
from .spectral import (
    extract_spectral_features, compute_beat_averaged_spectrum, SpectralBands,
)
from .pv_loops import (
    extract_pv_parameters, compute_ea, compute_single_beat_ees,
    calibrate_sv, compute_alpha_from_baseline, process_pv_loop,
)
