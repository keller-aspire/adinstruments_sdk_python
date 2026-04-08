import adi
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def export_comments(f, path):
    """
    Export comments from an ADI file to CSV.
    
    Parameters:
    -----------
    f : str or file object
        Path to ADI file or file object
    path : str
        Output path for CSV file
    """
    c = adi.extract_comments(f)
    c.to_csv(path)

def export_channels(f, path):
    """
    Export channel information from an ADI file to CSV.
    
    Parameters:
    -----------
    f : str or file object
        Path to ADI file or file object
    path : str
        Output path for CSV file
    """
    c = adi.extract_channels(f)
    c.to_csv(path)

def find_comments(comments, tags):
    """
    Find comments that match any of the specified tags.
    
    Parameters:
    -----------
    comments : pd.DataFrame
        DataFrame containing comments with a 'text' column
    tags : list of str
        List of tags/keywords to search for in comment text
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only comments that match the tags
        (case-insensitive matching)
    """
    # Build a single regex that matches any of the comment tags, case-insensitive
    pattern = "|".join(map(re.escape, tags))

    # Get a boolean mask of rows where 'text' matches the pattern (case-insensitive)
    mask = comments['text'].str.contains(pattern, case=False, na=False, regex=True)

    return comments[mask]

def get_nearby_events(time, comments, window_minutes):
    """
    Get comments within a time window around a specific timepoint.
    
    Parameters:
    -----------
    time : pd.Timestamp or datetime
        Center timepoint for the search window
    comments : pd.DataFrame
        DataFrame containing comments with a 'datetime' column
    window_minutes : float
        Total window size in minutes (e.g., 10 means ±5 minutes from time)
    
    Returns:
    --------
    pd.DataFrame
        Comments that fall within window_minutes/2 of the specified time
    """
    time_diff_minutes = (comments['datetime'] - time).abs().dt.total_seconds() / 60
    return comments[time_diff_minutes < (window_minutes / 2)]


def process_ekg(results_comment, sample_distance, minheight_peaks, minheight_troughs, bfilter):
    """
    Process EKG signal to detect peaks and troughs.
    
    Applies Savitzky-Golay filtering, optional butterworth filtering, 
    normalization, and peak/trough detection.
    
    Parameters:
    -----------
    results_comment : dict
        Dictionary containing 'data' key with DataFrame that includes 
        'ch3_EKG_mV' and 'relative_time' columns
    sample_distance : int
        Minimum distance between peaks/troughs in samples
    minheight_peaks : float
        Minimum normalized height for peak detection (0-1 scale)
    minheight_troughs : float
        Minimum normalized height for trough detection (0-1 scale)
    bfilter : tuple or None
        Butterworth filter coefficients (b, a) from butter(), or None to skip
    
    Returns:
    --------
    tuple
        (ekg_peak_data, ekg_trough_data, ekg_waveform, peaks, troughs) where:
        - ekg_peak_data: DataFrame with peak indices, times, and values
        - ekg_trough_data: DataFrame with trough indices, times, and values
        - ekg_waveform: Normalized (0-1) filtered EKG signal
        - peaks: Array of peak indices
        - troughs: Array of trough indices
    """
    raw_ekg = results_comment['data']['ch3_EKG_mV'].values
    
    # Apply Savitzky-Golay filter
    filtered_ekg = savgol_filter(raw_ekg, window_length=21, polyorder=3)
    
    # Apply optional Butterworth filter
    if bfilter is not None:
        b, a = bfilter
        filtered_ekg = filtfilt(b, a, filtered_ekg)
    
    # Normalize to 0-1 range
    ekg_waveform = (filtered_ekg - filtered_ekg.min()) / (filtered_ekg.max() - filtered_ekg.min())
    
    # Detect peaks
    peaks, _ = find_peaks(
        ekg_waveform,
        height=minheight_peaks,
        distance=sample_distance
    )
    
    ekg_peak_data = pd.DataFrame({
        'subset_index': peaks,
        'relative_time': results_comment['data'].iloc[peaks]['relative_time'].values,
        'ch3_EKG_mV': results_comment['data'].iloc[peaks]['ch3_EKG_mV'].values
    })
    
    # Detect troughs (by inverting the waveform)
    troughs, _ = find_peaks(
        1 - ekg_waveform,
        height=minheight_troughs,
        distance=sample_distance
    )
    
    ekg_trough_data = pd.DataFrame({
        'subset_index': troughs,
        'relative_time': results_comment['data'].iloc[troughs]['relative_time'].values,
        'ch3_EKG_mV': results_comment['data'].iloc[troughs]['ch3_EKG_mV'].values
    })
    
    return ekg_peak_data, ekg_trough_data, ekg_waveform, peaks, troughs

def calc_hr(ekg_peak_data, ekg_trough_data, ekg_peaks, ekg_troughs, 
            sampling_rate=1000, use_peaks=True, window_multiplier=1.1, verbose=True):
    """
    Calculate heart rate from EKG peaks or troughs and determine window size.
    
    Parameters:
    -----------
    ekg_peak_data : pd.DataFrame
        DataFrame with columns including 'relative_time' and datetime info for peaks
    ekg_trough_data : pd.DataFrame
        DataFrame with columns including 'relative_time' and datetime info for troughs
    ekg_peaks : np.ndarray
        Array of peak indices from find_peaks
    ekg_troughs : np.ndarray
        Array of trough indices from find_peaks
    sampling_rate : float, default=1000
        Sampling rate in Hz (samples per second)
    use_peaks : bool, default=True
        If True, use peaks for HR calculation; if False, use troughs
    window_multiplier : float, default=1.1
        Multiplier for window size (1.1 = 110% of one cardiac cycle)
    verbose : bool, default=True
        If True, print diagnostic information
    
    Returns:
    --------
    dict or None
        Dictionary containing HR metrics, or None if insufficient data.
        Keys include:
        - 'hr': Heart rate in bpm (from datetime intervals)
        - 'hr_peaks_index': HR calculated from peak indices (if available)
        - 'hr_troughs_index': HR calculated from trough indices (if available)
        - 'window': Recommended window size in samples for waveform analysis
        - 'method_used': 'peaks' or 'troughs'
        - 'cardiac_cycle_samples': Duration of one cardiac cycle in samples
        - 'cardiac_cycle_seconds': Duration of one cardiac cycle in seconds
        - 'n_peaks': Number of peaks detected
        - 'n_troughs': Number of troughs detected
    """
    
    # Select data based on use_peaks parameter
    if use_peaks:
        hr_intervals = ekg_peak_data['relative_time'].values
        method_used = 'peaks'
    else:
        hr_intervals = ekg_trough_data['relative_time'].values
        method_used = 'troughs'
    
    # Check for sufficient data
    if len(hr_intervals) < 2:
        if verbose:
            print(f"Insufficient {method_used} for HR calculation (need at least 2)")
        return None
    
    # Calculate HR from datetime/relative_time intervals
    sorted_intervals = np.sort(hr_intervals)
    interval_diffs = np.diff(sorted_intervals)  # differences in seconds
    mean_interval = np.mean(interval_diffs)
    
    # Convert to HR: (1 / interval_seconds) * 60 seconds/minute
    hr = (1 / mean_interval) * 60
    
    if verbose:
        print(f"HR ({method_used}): {hr:.2f} bpm")
    
    # HR from peaks (using indices)
    hr_peaks_index = None
    if len(ekg_peaks) > 1:
        peak_indices_sorted = np.sort(ekg_peaks)
        peak_index_diffs = np.diff(peak_indices_sorted)
        mean_peak_interval_samples = np.mean(peak_index_diffs)
        mean_peak_interval_seconds = mean_peak_interval_samples / sampling_rate
        hr_peaks_index = (1 / mean_peak_interval_seconds) * 60
        if verbose:
            print(f"HR-peaks (from indices): {hr_peaks_index:.2f} bpm")
    elif verbose:
        print("HR-peaks: insufficient peaks")
    
    # HR from troughs (using indices)
    hr_troughs_index = None
    if len(ekg_troughs) > 1:
        trough_indices_sorted = np.sort(ekg_troughs)
        trough_index_diffs = np.diff(trough_indices_sorted)
        mean_trough_interval_samples = np.mean(trough_index_diffs)
        mean_trough_interval_seconds = mean_trough_interval_samples / sampling_rate
        hr_troughs_index = (1 / mean_trough_interval_seconds) * 60
        if verbose:
            print(f"HR-troughs (from indices): {hr_troughs_index:.2f} bpm")
    elif verbose:
        print("HR-troughs: insufficient troughs")
    
    # Calculate window size
    cardiac_cycle_seconds = 60 / hr
    cardiac_cycle_samples = cardiac_cycle_seconds * sampling_rate
    window = cardiac_cycle_samples * window_multiplier
    
    # Round to nearest 10
    window = int(round(window / 10) * 10)
    
    if verbose:
        print(f"Window size: {window} samples ({window/sampling_rate:.3f} seconds)")
    
    return {
        'hr': hr,
        'hr_peaks_index': hr_peaks_index,
        'hr_troughs_index': hr_troughs_index,
        'window': window,
        'method_used': method_used,
        'cardiac_cycle_samples': cardiac_cycle_samples,
        'cardiac_cycle_seconds': cardiac_cycle_seconds,
        'n_peaks': len(ekg_peaks),
        'n_troughs': len(ekg_troughs)
    }

#####
# Interactive Plotting
#####

def visualize_window_plotly(results_comment, decimation_info=None):
    """
    Create interactive multi-channel waveform plot using Plotly.
    
    Parameters:
    -----------
    results_comment : dict
        Dictionary containing 'data' key with DataFrame that includes 
        channel columns (prefixed with 'ch') and 'relative_time' column
    decimation_info : dict, optional
        Dictionary containing decimation metadata with keys:
        - 'decimated': bool
        - 'original_hz': float
        - 'target_hz': float
        - 'final_points': int
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure with subplots for each channel, or error message figure
    """
    try:
        data = results_comment['data']
        ch_cols = [col for col in data.columns if col.startswith('ch')]
        
        if len(ch_cols) == 0:
            return go.Figure().add_annotation(
                text="No channel data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        n_channels = len(ch_cols)
        
        # Calculate appropriate height
        height_per_channel = 250
        total_height = max(400, height_per_channel * n_channels)
        
        fig = make_subplots(
            rows=n_channels, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=ch_cols
        )
        
        for i, channel in enumerate(ch_cols, 1):
            fig.add_trace(
                go.Scattergl(
                    x=data['relative_time'],
                    y=data[channel],
                    mode='lines',
                    name=channel,
                    line=dict(width=1),
                    hovertemplate='Time: %{x:.2f}s<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=i, col=1
            )
            
            fig.update_yaxes(title_text=channel, row=i, col=1)
        
        fig.update_xaxes(title_text="Relative Time (s)", row=n_channels, col=1)
        
        # Add decimation info to title if provided
        title_text = "Multi-Channel Waveforms"
        if decimation_info and decimation_info['decimated']:
            title_text += f" (Decimated: {decimation_info['original_hz']}Hz → {decimation_info['target_hz']}Hz, {decimation_info['final_points']} points)"
        
        fig.update_layout(
            height=total_height,
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=50, r=20, t=40, b=40),
            title=title_text
        )
        
        return fig
        
    except Exception as e:
        print(f"ERROR in visualize_window_plotly: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return go.Figure().add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )

def visualize_ekg_plotly(results_comment, ekg_peak_data, ekg_trough_data, ekg_waveform):
    """
    Create interactive EKG plot with detected peaks and troughs using Plotly.
    
    Parameters:
    -----------
    results_comment : dict
        Dictionary containing 'data' key with DataFrame that includes 
        'ch3_EKG_mV' and 'relative_time' columns
    ekg_peak_data : pd.DataFrame
        DataFrame with detected peaks containing 'relative_time' and 'ch3_EKG_mV' columns
    ekg_trough_data : pd.DataFrame
        DataFrame with detected troughs containing 'relative_time' and 'ch3_EKG_mV' columns
    ekg_waveform : np.ndarray
        Normalized (0-1) filtered EKG waveform from process_ekg()
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure showing raw EKG, filtered signal, and detected peaks/troughs
    """
    
    raw_ekg = results_comment['data']['ch3_EKG_mV'].values
    
    fig = go.Figure()
    
    # Raw EKG trace
    fig.add_trace(
        go.Scatter(
            x=results_comment['data']['relative_time'],
            y=results_comment['data']['ch3_EKG_mV'],
            mode='lines',
            name='Raw EKG',
            line=dict(color='blue', width=1),
            hovertemplate='Time: %{x:.2f}s<br>EKG: %{y:.2f} mV<extra></extra>'
        )
    )
    
    # Filtered EKG trace (scaled back to original range)
    ekg_scaled = ekg_waveform * (raw_ekg.max() - raw_ekg.min()) + raw_ekg.min()
    fig.add_trace(
        go.Scatter(
            x=results_comment['data']['relative_time'],
            y=ekg_scaled,
            mode='lines',
            name='Filtered',
            line=dict(color='red', width=1),
            opacity=0.3,
            hovertemplate='Time: %{x:.2f}s<br>Filtered: %{y:.2f} mV<extra></extra>'
        )
    )
    
    # Peak markers
    if len(ekg_peak_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=ekg_peak_data['relative_time'],
                y=ekg_peak_data['ch3_EKG_mV'],
                mode='markers',
                name='Peaks',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate='Peak<br>Time: %{x:.2f}s<br>Value: %{y:.2f} mV<extra></extra>'
            )
        )
    
    # Trough markers
    if len(ekg_trough_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=ekg_trough_data['relative_time'],
                y=ekg_trough_data['ch3_EKG_mV'],
                mode='markers',
                name='Troughs',
                marker=dict(color='darkgreen', size=8, symbol='triangle-down'),
                hovertemplate='Trough<br>Time: %{x:.2f}s<br>Value: %{y:.2f} mV<extra></extra>'
            )
        )
    
    fig.update_layout(
        title='EKG with Detected Peaks and Troughs',
        xaxis_title='Relative Time (s)',
        yaxis_title='EKG (mV)',
        height=500,
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig