"""Quick-start demo for the hemodynamics analysis package.

Usage:
    python demo_hemodynamics.py path/to/recording.adicht

Reads an .adicht file and runs the full analysis pipeline:
  1. Load ABP channel
  2. Continuous hemodynamics (MAP, HR, SBP, DBP, PP)
  3. Beat-by-beat feature extraction (14 features)
  4. Print results
"""

import sys
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_hemodynamics.py <file.adicht>")
        print("       python demo_hemodynamics.py <file.adicht> <channel_name>")
        sys.exit(1)

    filepath = sys.argv[1]
    channel = sys.argv[2] if len(sys.argv) > 2 else None

    # --- Load data ---
    import hemodynamics as hemo

    print(f"Loading {filepath}...")

    if filepath.endswith(".adicht"):
        channels = [channel] if channel else None
        data = hemo.load_adicht(filepath, channels=channels)
    elif filepath.endswith(".txt"):
        data = hemo.load_txt(filepath)
    else:
        data = hemo.load_auto(filepath)

    print(f"  Channels: {list(data['signals'].keys())}")
    print(f"  fs: {data['fs']} Hz")
    print(f"  Comments: {len(data['comments'])}")

    # Pick the first pressure channel available
    pressure_names = ['ABP', 'Pressure', 'ART', 'BP']
    signal_name = None
    for name in pressure_names:
        if name in data['signals']:
            signal_name = name
            break
    if signal_name is None:
        signal_name = list(data['signals'].keys())[0]

    abp = data['signals'][signal_name]
    fs = data['fs']
    print(f"\n  Using channel: {signal_name}")
    print(f"  Samples: {len(abp)}, Duration: {len(abp)/fs/60:.1f} min")
    print(f"  Range: {np.nanmin(abp):.1f} - {np.nanmax(abp):.1f}")

    # --- Continuous hemodynamics ---
    print("\n--- Continuous Hemodynamics (60s windows) ---")
    df = hemo.compute_continuous_hemodynamics(abp, fs)
    if len(df) > 0:
        print(df.to_string(index=False, float_format="%.1f"))
    else:
        print("  No valid windows.")

    # --- Feature extraction (first 60s) ---
    seg_len = min(len(abp), int(60 * fs))
    segment = abp[:seg_len]

    print(f"\n--- Beat Features (first {seg_len/fs:.0f}s) ---")
    features = hemo.extract_all_features(segment, fs)
    print(f"  Beats analyzed: {features['n_beats']}")

    if features['n_beats'] > 0:
        print(f"\n  {'Feature':<16s} {'Median':>10s} {'IQR':>10s}")
        print(f"  {'-'*38}")
        for k in hemo.FEATURE_NAMES:
            med = features['median'].get(k, np.nan)
            iqr = features['iqr'].get(k, np.nan)
            if np.isnan(med):
                print(f"  {k:<16s} {'NaN':>10s} {'NaN':>10s}")
            else:
                print(f"  {k:<16s} {med:10.2f} {iqr:10.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
