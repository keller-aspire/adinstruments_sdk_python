"""Verify the range check fix on Ian's file."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import adi
from hemodynamics.continuous import compute_continuous_hemodynamics
from hemodynamics.features import extract_all_features

FILEPATH = r"C:\Users\caomi\Downloads\20250612-ALI7-N3791.adicht"

f = adi.read_file(FILEPATH)
abp_ch = [c for c in f.channels if c.name.strip() == 'ABP'][0]

for r in [2, 3, 4, 5]:
    data = abp_ch.get_data(r).astype(np.float64)
    seg = data[:60000]

    df = compute_continuous_hemodynamics(seg, 1000.0)
    n_valid = df['MAP'].notna().sum()

    if n_valid > 0:
        row = df.iloc[0]
        print(f"R{r}: MAP={row['MAP']:.1f}  HR={row['HR']:.1f}  "
              f"SBP={row['SBP']:.1f}  DBP={row['DBP']:.1f}")
    else:
        print(f"R{r}: continuous hemo = NaN")

    feat = extract_all_features(seg, 1000.0)
    mdpdt = feat['median'].get('maxDPDT', np.nan)
    print(f"     {feat['n_beats']} beats, maxDPDT={mdpdt:.1f}")
    print()
