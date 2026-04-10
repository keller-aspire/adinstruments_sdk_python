"""Debug Ian's file — check all records to find the real data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import adi
from hemodynamics.cycles import detect_peaks
from hemodynamics.continuous import compute_continuous_hemodynamics

FILEPATH = r"C:\Users\caomi\Downloads\20250612-ALI7-N3791.adicht"

f = adi.read_file(FILEPATH)
print(f"Channels: {f.n_channels}, Records: {f.n_records}\n")

# Check ABP across all records
abp_ch = None
for i in range(f.n_channels):
    if f.channels[i].name.strip() == 'ABP':
        abp_ch = f.channels[i]
        break

if abp_ch is None:
    print("No ABP channel found!")
    sys.exit(1)

for r in range(1, f.n_records + 1):
    n = abp_ch.n_samples[r - 1]
    if n == 0:
        print(f"Record {r}: empty")
        continue

    data = abp_ch.get_data(r)
    med = np.nanmedian(data)
    mn, mx = np.nanmin(data), np.nanmax(data)
    dur = n / abp_ch.fs[r - 1] / 60
    print(f"Record {r}: {n} samples ({dur:.1f} min), "
          f"ABP median={med:.1f}, range=[{mn:.1f}, {mx:.1f}]")

    # Try peak detection on first 60s
    seg = data[:min(len(data), 60000)]
    vals, locs = detect_peaks(seg, abp_ch.fs[r - 1])
    print(f"  Peaks (first 60s): {len(locs)}")

    if len(locs) >= 3:
        hr_approx = len(locs) / (len(seg) / abp_ch.fs[r - 1]) * 60
        print(f"  HR ~ {hr_approx:.0f} bpm")

        df = compute_continuous_hemodynamics(seg, abp_ch.fs[r - 1])
        n_valid = df['MAP'].notna().sum()
        if n_valid > 0:
            row = df.iloc[0]
            print(f"  MAP={row['MAP']:.1f}, HR={row['HR']:.1f}, "
                  f"SBP={row['SBP']:.1f}, DBP={row['DBP']:.1f}")
        else:
            print(f"  Continuous hemo: all NaN")

            # Diagnose why
            clean = seg[~np.isnan(seg)]
            sig_range = float(clean.max() - clean.min())
            print(f"    Signal range check: {sig_range:.1f} "
                  f"(need 10-300 to pass)")
    print()

print("Done.")
