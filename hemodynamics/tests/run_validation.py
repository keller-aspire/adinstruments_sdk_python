"""Standalone validation script — run directly with python."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from hemodynamics.io import load_txt
from hemodynamics.features import extract_all_features
from hemodynamics.continuous import compute_continuous_hemodynamics
from hemodynamics.cycles import detect_peaks
from hemodynamics.normalization import average_beats

N346 = r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project\SwineHemorrhage\Data_txt_1000Hz\Group1\N346.txt"

print("Loading N346...")
data = load_txt(N346)
abp = data['signals']['ABP']
fs = data['fs']
print(f"  {len(abp)} samples, fs={fs}, duration={len(abp)/fs/60:.1f} min")
print(f"  ABP median: {np.nanmedian(abp):.1f} mmHg")
print(f"  Comments: {len(data['comments'])}")
for c in data['comments'][:15]:
    print(f"    {c['time_s']/60:.1f}min: {c['text']}")

# First 60s
seg = abp[:60000]
print(f"\nSegment [0:60s]: range={np.nanmin(seg):.1f}-{np.nanmax(seg):.1f}")

_, locs = detect_peaks(seg, fs)
print(f"  Peaks: {len(locs)}, HR~{len(locs)/60*60:.0f} bpm")

df = compute_continuous_hemodynamics(seg, fs)
if len(df) > 0:
    r = df.iloc[0]
    print(f"  MAP={r['MAP']:.1f}, HR={r['HR']:.1f}, SBP={r['SBP']:.1f}, DBP={r['DBP']:.1f}")

result = average_beats(seg, fs, n_points=100)
if result['mean'] is not None:
    print(f"  AvgBeat: {result['n_cycles']} cycles")

features = extract_all_features(seg, fs)
print(f"\nFeatures ({features['n_beats']} beats):")
for k, v in features['median'].items():
    print(f"  {k:15s} = {v:.4f}" if not np.isnan(v) else f"  {k:15s} = NaN")

print("\nDone.")
