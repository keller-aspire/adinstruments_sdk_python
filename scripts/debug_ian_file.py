"""Debug Ian's test file to identify NaN issue."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import adi

FILEPATH = r"C:\Users\caomi\Downloads\20250612-ALI7-N3791.adicht"

# Step 1: Inspect file structure
f = adi.read_file(FILEPATH)
print(f"File: {FILEPATH}")
print(f"Channels: {f.n_channels}, Records: {f.n_records}")

print("\nChannel listing:")
for i in range(f.n_channels):
    ch = f.channels[i]
    name = ch.name
    units = ch.units[0] if ch.units else "?"
    fs = ch.fs[0] if ch.fs else "?"
    n = ch.n_samples[0] if ch.n_samples else 0
    print(f"  [{i}] {name:25s} units={units:10s} fs={fs}  samples_r1={n}")

# Step 2: Try loading with hemodynamics
from hemodynamics.io import load_adicht
from hemodynamics.continuous import compute_continuous_hemodynamics
from hemodynamics.cycles import detect_peaks, CycleDetectionParams
from hemodynamics.features import extract_all_features

# Find a pressure-like channel
print("\n--- Attempting hemodynamics analysis ---")
data = load_adicht(FILEPATH, record=1)
print(f"Loaded channels: {list(data['signals'].keys())}")
print(f"fs: {data['fs']}")

for ch_name, sig in data['signals'].items():
    if len(sig) == 0:
        continue
    med = np.nanmedian(sig)
    rng = np.nanmax(sig) - np.nanmin(sig)
    print(f"\n  {ch_name}: {len(sig)} samples, median={med:.1f}, range={rng:.1f}")

    # Try peak detection on channels that look like pressure
    if rng > 10 and med > 10:
        vals, locs = detect_peaks(sig[:60000], data['fs'])
        print(f"    Peak detection (default ABP): {len(locs)} peaks in first 60s")

        if len(locs) < 3:
            # Try with LVP preset
            vals2, locs2 = detect_peaks(sig[:60000], data['fs'],
                                        CycleDetectionParams.lvp_default())
            print(f"    Peak detection (LVP preset): {len(locs2)} peaks")

            # Try with very permissive params
            vals3, locs3 = detect_peaks(sig[:60000], data['fs'],
                                        CycleDetectionParams(
                                            smooth_window_ms=50,
                                            min_distance_ms=200,
                                            min_prominence_mmhg=3,
                                            min_height_mmhg=None))
            print(f"    Peak detection (permissive):  {len(locs3)} peaks")

        # Try continuous hemodynamics
        df = compute_continuous_hemodynamics(sig[:60000], data['fs'])
        n_valid = df['MAP'].notna().sum()
        print(f"    Continuous hemo: {len(df)} windows, {n_valid} valid")
        if n_valid > 0:
            print(f"      MAP={df['MAP'].dropna().iloc[0]:.1f}, "
                  f"HR={df['HR'].dropna().iloc[0]:.1f}")

print("\nDone.")
