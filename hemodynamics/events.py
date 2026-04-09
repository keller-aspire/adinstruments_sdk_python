"""Comment-driven segmentation and phase landmark extraction.

Ports:
  - standardize_comment() from preprocess_labchart.py
  - _get_phase_landmarks() from analyze_pv_loops.py
  - extract_comment_based_segment.m / extract_time_calculated_segment.m
"""

import re
import numpy as np


def standardize_comment(raw_text):
    """Normalize raw comment text to canonical protocol labels.

    Returns a list of standardized strings. Returns [] to drop the comment.
    Ported from preprocess_labchart.py.
    """
    text = raw_text.strip()
    if not text or text == "[]":
        return []

    # Special: combined "Baseline End HS (Fast) Start"
    if re.search(r"baseline\s+end.*hs.*fast.*start", text, re.I):
        return ["Baseline End", "HS Fast Start"]

    # Special: combined "START RES 500ML BLOOD"
    if re.search(r"start\s+res.*blood", text, re.I):
        return ["Resuscitation Start", "Blood Start"]

    # Protocol phases
    rules = [
        (r"^(start\s+)?b(ase)?l(ine)?(\s+start)?$", "Baseline Start"),
        (r"^(end\s+)?b(ase)?l(ine)?(\s+end)?$", "Baseline End"),
        (r"hs\s*\(?\s*fast\s*\)?\s*start|start\s+fast\s+hs", "HS Fast Start"),
        (r"hs\s*\(?\s*fast\s*\)?\s*end|end\s+fast\s+hs", "HS Fast End"),
        (r"hs\s*\(?\s*slow\s*\)?\s*start|start\s+slow\s+hs", "HS Slow Start"),
        (r"hs\s*\(?\s*slow\s*\)?\s*end|end\s+(of\s+)?(slow\s+)?hs", "HS Slow End"),
        (r"^(start\s+)?delay(\s+start)?$", "Delay Start"),
        (r"end.*delay|delay.*end", "Delay End"),
        (r"start\s+res", "Resuscitation Start"),
        (r"study\s+end|end\s+of\s+study", "Study End"),
    ]
    for pattern, label in rules:
        if re.search(pattern, text, re.I):
            return [label]

    # Post time points (1H-6H)
    m = re.match(r"^([1-6])\s*h(our|r)?\s*(post)?\s*$", text, re.I)
    if m:
        return [f"{m.group(1)}H Post"]

    # CO values
    m = re.match(r"^co\s*(\d+\.{0,2}\d+)$", text, re.I)
    if m:
        val = m.group(1).replace("..", ".")
        return [f"CO {val}"]

    # PEEP
    if re.search(r"peep\s*(15\s*)?(start|st$)|start\s*peep|^peep\s*15$|^peep$",
                 text, re.I):
        return ["PEEP Start"]
    if re.search(r"peep\s*(15\s*)?end|end\s*peep", text, re.I):
        return ["PEEP End"]

    # NE + VASO combined
    m = re.match(
        r"^ne\s*(?:on\s+)?(\d+\.?\d*)[,\s]+"
        r"(v[ae]s[eo]|vaso|vp)\s*(\d+\.?\d*)", text, re.I)
    if m:
        return [f"NE {m.group(1)} VASO {m.group(3)}"]

    # NE + VASO OFF
    m = re.match(
        r"^ne\s*(?:on\s+)?(\d+\.?\d*)\s+(v[ae]s[eo]|vaso|vp)\s+off",
        text, re.I)
    if m:
        return [f"NE {m.group(1)} VASO Off"]

    # VASO ... NE (reversed)
    m = re.match(r"^v[ae]s[eo]\s*(\d+\.?\d*)\s+ne\s*(\d+\.?\d*)", text, re.I)
    if m:
        return [f"NE {m.group(2)} VASO {m.group(1)}"]

    # NE only
    m = re.match(r"^ne\s*(?:on\s+)?(\d+\.?\d*)$", text, re.I)
    if m:
        return [f"NE {m.group(1)}"]

    # Reversed NE
    m = re.match(r"^(\d+\.?\d*)\s+ne$", text, re.I)
    if m:
        return [f"NE {m.group(1)}"]

    # NE off/stopped
    if re.search(r"^ne\s+(off|stopped)", text, re.I):
        return ["NE Off"]

    # NE bolus
    if re.search(r"^ne\s+bolus", text, re.I):
        return ["NE Bolus"]

    # EPI
    m = re.match(r"^epi\s*(\d+\.?\d*)", text, re.I)
    if m:
        return [f"EPI {m.group(1)}"]

    # Fluid boluses
    if re.search(r"start.*(blood|prbc)|(blood|prbc).*start", text, re.I):
        return ["Blood Start"]
    if re.search(r"end.*(blood|prbc)|(blood|prbc).*(done|end)", text, re.I):
        return ["Blood End"]
    if re.search(r"albumin.*start|start.*albumin", text, re.I):
        return ["Albumin Start"]
    if re.search(r"albumin.*(end|done)|end.*albumin", text, re.I):
        return ["Albumin End"]
    if re.search(r"start.*(lr|500)|^lr\s*500\s*(start|cc)?$", text, re.I):
        return ["LR 500 Start"]
    if re.search(r"en[dr].*(lr|500)|(lr|500).*(done|end|in$)", text, re.I):
        return ["LR 500 End"]

    # Dextrose
    if re.search(r"d5[0w]?\b", text, re.I):
        return ["Dextrose"]

    return []


def get_phase_landmarks(comments, phase_definitions=None):
    """Extract protocol phase landmarks from comment list.

    Parameters
    ----------
    comments : list of dict
        Each with 'text' (str), 'time_s' (float), and optionally 'record' (int).
    phase_definitions : dict, optional
        Override mapping from comment text to phase_code.

    Returns
    -------
    dict mapping phase_code -> {'record': int, 'time_s': float}
    """
    landmarks = {}
    bs_start = bs_end = None

    for c in comments:
        text = c['text']
        time_s = c['time_s']
        record = c.get('record', 1)

        if phase_definitions and text in phase_definitions:
            code = phase_definitions[text]
            landmarks[code] = {'record': record, 'time_s': time_s}
            continue

        if text == "Baseline Start":
            bs_start = {'record': record, 'time_s': time_s}
        elif text == "Baseline End":
            bs_end = {'record': record, 'time_s': time_s}
        elif text == "HS Slow End":
            landmarks["EH"] = {'record': record, 'time_s': time_s}
        elif text == "HS Fast End":
            landmarks.setdefault("EH", {'record': record, 'time_s': time_s})
        elif text == "Delay End":
            landmarks["ED"] = {'record': record, 'time_s': time_s}
        elif text == "Resuscitation Start":
            landmarks["AR"] = {'record': record, 'time_s': time_s + 120}
        elif text == "Study End":
            landmarks["SE"] = {'record': record, 'time_s': time_s}
        else:
            m = re.match(r"^(\d)H Post$", text)
            if m:
                landmarks[f"H{m.group(1)}"] = {
                    'record': record, 'time_s': time_s}

    # Baseline: midpoint if both markers exist
    if bs_start and bs_end and bs_start['record'] == bs_end['record']:
        landmarks["BS"] = {
            'record': bs_start['record'],
            'time_s': (bs_start['time_s'] + bs_end['time_s']) / 2,
        }
    elif bs_start:
        landmarks["BS"] = {
            'record': bs_start['record'],
            'time_s': bs_start['time_s'] + 60,
        }

    return landmarks


def extract_segment(signal, fs, center_time_s, window_s=60.0, offset_s=0.0):
    """Extract a segment of signal around a time point.

    Parameters
    ----------
    signal : np.ndarray
    fs : float
    center_time_s : float
        Center time in seconds.
    window_s : float
        Window duration in seconds.
    offset_s : float
        Offset from center (positive = later).

    Returns
    -------
    np.ndarray (the extracted segment, or empty if out of bounds)
    """
    center_sample = int((center_time_s + offset_s) * fs)
    half_window = int(window_s / 2 * fs)

    start = max(0, center_sample - half_window)
    end = min(len(signal), center_sample + half_window)

    if end - start < int(fs * 5):  # minimum 5 seconds
        return np.array([])

    return signal[start:end].copy()


def find_best_window(signal, fs, center_time_s,
                     scan_range_s=60.0, scan_step_s=1.0,
                     window_s=5.0, score_fn=None, volume=None):
    """Scan for the highest-quality window near a landmark.

    Ports the scanning logic from analyze_pv_loops.py extract_pv_waveforms().

    Parameters
    ----------
    signal : np.ndarray
        Pressure waveform.
    fs : float
    center_time_s : float
        Landmark time to scan around.
    scan_range_s : float
        Search +/- this many seconds around center.
    scan_step_s : float
        Step size in seconds.
    window_s : float
        Extraction window duration.
    score_fn : callable, optional
        Function(segment, fs) -> float. Defaults to cycles.score_quality.
    volume : np.ndarray, optional
        Volume signal (same length/fs as signal).

    Returns
    -------
    dict with 'start_sample', 'end_sample', 'score', 'signal_segment',
    and optionally 'volume_segment'.
    """
    from .cycles import score_quality as default_score_fn

    if score_fn is None:
        score_fn = None  # use default below

    window_samples = int(window_s * fs)
    best = {'start_sample': 0, 'end_sample': 0, 'score': 0.0,
            'signal_segment': np.array([]), 'time_s': center_time_s}

    t_start = center_time_s - scan_range_s
    t_end = center_time_s + scan_range_s

    t = t_start
    while t <= t_end:
        s_idx = int(t * fs)
        e_idx = s_idx + window_samples

        if s_idx < 0 or e_idx > len(signal):
            t += scan_step_s
            continue

        seg_p = signal[s_idx:e_idx]
        seg_v = volume[s_idx:e_idx] if volume is not None else None

        if score_fn is not None:
            score = score_fn(seg_p, fs)
        else:
            result = default_score_fn(seg_p, fs, volume=seg_v)
            score = result['score']

        if score > best['score']:
            best = {
                'start_sample': s_idx,
                'end_sample': e_idx,
                'score': score,
                'signal_segment': seg_p.copy(),
                'time_s': t,
            }
            if seg_v is not None:
                best['volume_segment'] = seg_v.copy()

        t += scan_step_s

    return best
