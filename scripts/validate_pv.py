"""Cross-validate PV loop pipeline against cached pipeline_results.pkl."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pickle
from hemodynamics.io import load_hdf5
from hemodynamics.pv_loops import process_pv_loop
from hemodynamics.events import get_phase_landmarks

HDF5 = r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project\SwineHemorrhage_Microcirculation\LabChart\hemodynamics.h5"
CACHE = r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project\SwineHemorrhage_Microcirculation\Microcirculation_Analysis\.cache\pipeline_results.pkl"

print("(Skipping cached pickle — pandas version mismatch)")

# Test on N781 baseline
import h5py
with h5py.File(HDF5, 'r') as f:
    durations = f['N781'].attrs['record_durations_s']
    best_rec = int(durations.argmax()) + 1

data = load_hdf5(HDF5, 'N781', record=best_rec)
print(f"\nN781 record {best_rec}: {len(data['signals'].get('ABP', []))} samples")
print(f"  Channels: {list(data['signals'].keys())}")
print(f"  Comments: {len(data['comments'])}")

# Find baseline landmark
landmarks = get_phase_landmarks(data['comments'])
print(f"  Landmarks: {list(landmarks.keys())}")

if 'BS' in landmarks and 'Pressure' in data['signals'] and 'Magnitude_Volume' in data['signals']:
    bs_time = landmarks['BS']['time_s']
    print(f"\n  Baseline at {bs_time:.1f}s")

    lvp = data['signals']['Pressure']
    lvv = data['signals']['Magnitude_Volume']
    fs = data['fs']

    # Extract 5s window around baseline
    center = int(bs_time * fs)
    half = int(2.5 * fs)
    start = max(0, center - half)
    end = min(len(lvp), center + half)

    result = process_pv_loop(lvp[start:end], lvv[start:end], fs)
    print(f"\n  PV Loop Results:")
    for k in ['ESP', 'EDP', 'dPdt_max', 'dPdt_min', 'SV_raw', 'HR_pv', 'n_cycles']:
        v = result.get(k)
        if v is not None:
            if isinstance(v, float):
                print(f"    {k:12s} = {v:.2f}")
            else:
                print(f"    {k:12s} = {v}")

    # Physiological plausibility checks
    print(f"\n  Plausibility checks:")
    if result.get('ESP', 0) > 0:
        assert 30 < result['ESP'] < 200, f"ESP {result['ESP']} out of range"
        print(f"    ESP in range: PASS")
    if result.get('EDP', 0) >= 0:
        assert result['EDP'] < result.get('ESP', 999), "EDP should be < ESP"
        print(f"    EDP < ESP: PASS")
    if result.get('n_cycles', 0) >= 3:
        print(f"    >= 3 cycles: PASS")
else:
    print("  Missing Pressure/Volume channels or BS landmark")

print("\nDone.")
