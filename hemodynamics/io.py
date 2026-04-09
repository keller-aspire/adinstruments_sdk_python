"""Data loading abstraction for LabChart data.

Provides a uniform interface to load waveform data from:
  - .adicht files (via the adi SDK)
  - .txt LabChart exports (tab-delimited with optional header)
  - .h5 preprocessed HDF5 files

All loaders return a dict with a consistent schema:
    {
        'signals': {channel_name: np.ndarray, ...},
        'fs': float,
        'comments': [{'text': str, 'time_s': float}, ...],
        'metadata': {...}
    }
"""

import os
import re
import numpy as np

# Column name presets for common LabChart export layouts
COLUMN_PRESETS = {
    3: ['Time', 'ABP', 'VBP'],
    6: ['Time', 'ABP', 'VBP', 'PAP', 'LVV', 'LVP'],
}


def _detect_header_lines(filepath):
    """Auto-detect header line count by finding the first numeric data line."""
    header_lines = 0
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                header_lines += 1
                continue
            if re.match(r'^\d', stripped):
                break
            header_lines += 1
    return header_lines


def _parse_header(filepath, n_header_lines):
    """Parse LabChart .txt header for metadata (sampling interval, channel info)."""
    meta = {}
    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= n_header_lines:
                break
            line = line.strip()
            if line.startswith('Interval='):
                parts = line.split('\t')
                if len(parts) >= 2:
                    # e.g. "0.001 s"
                    val = parts[1].strip().split()[0]
                    try:
                        meta['dt'] = float(val)
                    except ValueError:
                        pass
            elif line.startswith('ChannelTitle='):
                parts = line.split('\t')
                meta['channel_titles'] = [p.strip() for p in parts[1:] if p.strip()]
            elif line.startswith('UnitName='):
                parts = line.split('\t')
                meta['units'] = [p.strip() for p in parts[1:] if p.strip()]
            elif line.startswith('ExcelDateTime='):
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        meta['excel_datetime'] = float(parts[1].strip())
                    except ValueError:
                        pass
    return meta


def _detect_n_columns(filepath, header_lines):
    """Detect the number of numeric columns from the first data line."""
    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i < header_lines:
                continue
            parts = line.strip().split('\t')
            n_numeric = 0
            for p in parts:
                try:
                    float(p)
                    n_numeric += 1
                except (ValueError, TypeError):
                    break  # stop at first non-numeric (comment column)
            return n_numeric
    return 3  # fallback


def load_txt(filepath, column_names=None, fs=None):
    """Load waveform data from a LabChart .txt export.

    Handles both header-bearing files (Group1/2/3) and headerless files
    (Advanced). Auto-detects header lines and column count.

    Parameters
    ----------
    filepath : str
        Path to the .txt file.
    column_names : list[str], optional
        Column names. If None, auto-detected from header or preset.
    fs : float, optional
        Sampling frequency in Hz. If None, inferred from header or
        from time column differences.

    Returns
    -------
    dict with keys: signals, fs, comments, metadata
    """
    filepath = str(filepath)

    # Detect header and columns
    n_header = _detect_header_lines(filepath)
    header_meta = _parse_header(filepath, n_header) if n_header > 0 else {}
    n_cols = _detect_n_columns(filepath, n_header)

    # Determine column names
    if column_names is None:
        if 'channel_titles' in header_meta:
            column_names = ['Time'] + header_meta['channel_titles']
        elif n_cols in COLUMN_PRESETS:
            column_names = COLUMN_PRESETS[n_cols]
        else:
            column_names = ['Time'] + [f'Ch{i}' for i in range(1, n_cols)]

    # Ensure column_names length matches
    while len(column_names) < n_cols:
        column_names.append(f'Ch{len(column_names)}')

    # Read numeric data — read all columns as string first, parse carefully
    # This handles the trailing comment column gracefully
    raw_lines = []
    comments_list = []

    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if i < n_header:
                continue
            parts = line.rstrip('\n\r').split('\t')

            # Parse numeric columns
            nums = []
            comment_text = ''
            for j, p in enumerate(parts):
                if j < n_cols:
                    try:
                        nums.append(float(p))
                    except (ValueError, TypeError):
                        nums.append(np.nan)
                else:
                    # Remaining parts are comment text
                    comment_text = '\t'.join(parts[n_cols:]).strip()
                    # Clean comment markers
                    comment_text = comment_text.replace('#', '').replace('*', '').strip()
                    break

            if len(nums) == n_cols:
                raw_lines.append(nums)
                if comment_text:
                    # time_s relative to first sample
                    comments_list.append({
                        'text': comment_text,
                        'sample_idx': len(raw_lines) - 1,
                    })

    if not raw_lines:
        raise ValueError(f"No data found in {filepath}")

    data = np.array(raw_lines, dtype=np.float64)

    # Determine sampling frequency
    if fs is not None:
        pass
    elif 'dt' in header_meta:
        fs = 1.0 / header_meta['dt']
    else:
        # Infer from time column differences
        time_col = data[:min(100, len(data)), 0]
        dt_vals = np.diff(time_col)
        median_dt = np.median(dt_vals)
        if median_dt > 0:
            # Time column is Excel datetime (days), convert to seconds
            dt_seconds = median_dt * 86400.0
            fs = round(1.0 / dt_seconds)
        else:
            fs = 1000.0  # fallback

    # Build signals dict (skip Time column)
    signals = {}
    for col_idx in range(1, n_cols):
        name = column_names[col_idx] if col_idx < len(column_names) else f'Ch{col_idx}'
        signals[name] = data[:, col_idx]

    # Convert comment sample indices to time_s
    comments = []
    for c in comments_list:
        comments.append({
            'text': c['text'],
            'time_s': c['sample_idx'] / fs,
        })

    return {
        'signals': signals,
        'fs': float(fs),
        'comments': comments,
        'metadata': {
            'source': 'txt',
            'filepath': filepath,
            'n_header_lines': n_header,
            'n_columns': n_cols,
            'column_names': column_names,
            'n_samples': len(data),
            **header_meta,
        },
    }


def load_adicht(filepath, channels=None, record=1):
    """Load waveform data from an .adicht file via the adi SDK.

    Parameters
    ----------
    filepath : str
        Path to the .adicht file.
    channels : list[str], optional
        Channel names to load. If None, loads all channels.
    record : int
        Record number (1-indexed). Default: 1.

    Returns
    -------
    dict with keys: signals, fs, comments, metadata
    """
    import adi

    filepath = str(filepath)
    f = adi.read_file(filepath)

    # Build channel name mapping
    ch_map = {}
    for i in range(f.n_channels):
        name = f.channels[i].name.strip()
        # Clean to match standard names
        clean = re.sub(r'[^A-Za-z0-9_]', '_', name)
        clean = re.sub(r'_+', '_', clean).strip('_')
        ch_map[clean] = (i, f.channels[i])

    # Select channels
    if channels is None:
        selected = list(ch_map.keys())
    else:
        selected = [c for c in channels if c in ch_map]

    if not selected:
        raise ValueError(
            f"No matching channels found. Available: {list(ch_map.keys())}")

    # Read signals
    signals = {}
    fs = None
    CHUNK_SIZE = 2_000_000

    for ch_name in selected:
        idx, ch_obj = ch_map[ch_name]
        n_samples = ch_obj.n_samples[record - 1]

        if fs is None:
            fs = ch_obj.fs[record - 1]

        if n_samples == 0:
            signals[ch_name] = np.array([], dtype=np.float64)
            continue

        # Read with chunking for large records
        if n_samples <= CHUNK_SIZE:
            data = ch_obj.get_data(record)
        else:
            chunks = []
            pos = 1
            while pos <= n_samples:
                end = min(pos + CHUNK_SIZE - 1, n_samples)
                chunks.append(ch_obj.get_data(record, start_sample=pos,
                                              stop_sample=end))
                pos = end + 1
            data = np.concatenate(chunks)

        signals[ch_name] = data.astype(np.float64)

    # Extract comments
    comments = []
    try:
        comments_df = adi.extract_comments(f)
        for _, row in comments_df.iterrows():
            if int(row['record_id']) == record:
                comments.append({
                    'text': str(row['text']).strip(),
                    'time_s': float(row['time']),
                })
    except Exception:
        pass

    # Channel units
    units = {}
    for ch_name in selected:
        idx, ch_obj = ch_map[ch_name]
        try:
            units[ch_name] = ch_obj.units[record - 1]
        except (IndexError, AttributeError):
            units[ch_name] = ''

    return {
        'signals': signals,
        'fs': float(fs) if fs else 1000.0,
        'comments': comments,
        'metadata': {
            'source': 'adicht',
            'filepath': filepath,
            'record': record,
            'n_records': f.n_records,
            'n_channels_total': f.n_channels,
            'channel_names': selected,
            'units': units,
        },
    }


def load_hdf5(filepath, animal, channels=None, record=1):
    """Load waveform data from a preprocessed HDF5 file.

    Expects the structure produced by preprocess_labchart.py:
        /{animal}/R{n}/{channel_name}  (float32 datasets)
        /{animal}/comments/text, time_s, record  (comment datasets)
        /{animal}.attrs: fs, channel_names, channel_units, ...

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    animal : str
        Animal ID (e.g. 'N781').
    channels : list[str], optional
        Channel names to load. If None, loads all available.
    record : int
        Record number (1-indexed). Default: 1.

    Returns
    -------
    dict with keys: signals, fs, comments, metadata
    """
    import h5py

    filepath = str(filepath)

    with h5py.File(filepath, 'r') as hf:
        if animal not in hf:
            raise KeyError(
                f"Animal '{animal}' not found. Available: {list(hf.keys())}")

        grp = hf[animal]
        fs = float(grp.attrs['fs'])
        available_channels = list(grp.attrs['channel_names'])
        channel_units = list(grp.attrs.get('channel_units', []))

        rec_key = f'R{record}'
        if rec_key not in grp:
            raise KeyError(
                f"Record '{rec_key}' not found for {animal}. "
                f"Available: {[k for k in grp.keys() if k.startswith('R')]}")

        rec_grp = grp[rec_key]

        # Select channels
        if channels is None:
            selected = [c for c in available_channels if c in rec_grp]
        else:
            selected = [c for c in channels if c in rec_grp]

        signals = {}
        for ch_name in selected:
            signals[ch_name] = rec_grp[ch_name][:].astype(np.float64)

        # Load comments for this record
        comments = []
        if 'comments' in grp:
            cgrp = grp['comments']
            texts = cgrp['text'][:]
            times = cgrp['time_s'][:]
            records = cgrp['record'][:]
            for t, time_s, rec_id in zip(texts, times, records):
                if int(rec_id) == record:
                    text = t.decode('utf-8') if isinstance(t, bytes) else str(t)
                    comments.append({
                        'text': text,
                        'time_s': float(time_s),
                    })

        # Units mapping
        units = {}
        for i, ch in enumerate(available_channels):
            if ch in selected and i < len(channel_units):
                u = channel_units[i]
                units[ch] = u.decode('utf-8') if isinstance(u, bytes) else str(u)

    return {
        'signals': signals,
        'fs': fs,
        'comments': comments,
        'metadata': {
            'source': 'hdf5',
            'filepath': filepath,
            'animal': animal,
            'record': record,
            'channel_names': selected,
            'units': units,
        },
    }


def load_auto(filepath, channel=None, **kwargs):
    """Auto-detect file format and load.

    Parameters
    ----------
    filepath : str
        Path to data file (.adicht, .txt, or .h5).
    channel : str, optional
        For .adicht/.h5: specific channel to load (wraps in list).
        For .txt: ignored (all columns loaded).
    **kwargs
        Passed to the appropriate loader.

    Returns
    -------
    dict with keys: signals, fs, comments, metadata
    """
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.adicht':
        channels = [channel] if channel else None
        return load_adicht(filepath, channels=channels, **kwargs)
    elif ext in ('.h5', '.hdf5'):
        channels = [channel] if channel else None
        return load_hdf5(filepath, channels=channels, **kwargs)
    elif ext == '.txt':
        return load_txt(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
