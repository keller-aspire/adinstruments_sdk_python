"""Cross-validate against MATLAB Baseline End results for N346."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from hemodynamics.io import load_txt
from hemodynamics.features import extract_all_features
from hemodynamics.continuous import compute_continuous_hemodynamics
from hemodynamics.events import standardize_comment

TXT = r"C:\Users\caomi\OneDrive - Johns Hopkins\C_Side_Project\SwineHemorrhage\Data_txt_1000Hz\Group1\N346.txt"

data = load_txt(TXT)
abp = data['signals']['ABP']
fs = data['fs']

# Find Baseline End
bl_end_s = None
for c in data['comments']:
    std = standardize_comment(c['text'])
    if 'Baseline End' in std:
        bl_end_s = c['time_s']
        break

print(f"Baseline End at {bl_end_s:.1f}s ({bl_end_s/60:.1f} min)")

# 60s before Baseline End
end_sample = int(bl_end_s * fs)
start_sample = max(0, end_sample - 60000)
seg = abp[start_sample:end_sample]
print(f"Segment: {start_sample}-{end_sample} ({len(seg)} samples)")

# Continuous hemodynamics
df = compute_continuous_hemodynamics(seg, fs)
if len(df) > 0:
    r = df.iloc[0]
    print(f"\nHemodynamics (our pipeline):")
    print(f"  SBP={r['SBP']:.1f}, DBP={r['DBP']:.1f}, MAP={r['MAP']:.1f}, HR={r['HR']:.1f}")

print(f"\nMATLAB reference (Baseline End, N346):")
print(f"  SBP_median=124.69, DBP_median=75.39, MAP_median=92.55")

# Features
features = extract_all_features(seg, fs)
print(f"\nFeatures ({features['n_beats']} beats):")
med = features['median']
for k, v in med.items():
    if np.isnan(v):
        print(f"  {k:15s} = NaN")
    else:
        print(f"  {k:15s} = {v:.4f}")

# MATLAB comparison
print(f"\nKey comparisons:")
print(f"  SRT (ms):          ours={med['SRT']:.1f}   MATLAB upstroke_time_median=167.0 ms")
print(f"  tau (s):           ours={med['tau']:.4f}")
print(f"  DiastolicR2:       ours={med['DiastolicR2']:.4f}")
