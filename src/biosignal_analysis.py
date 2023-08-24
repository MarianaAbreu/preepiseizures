# built-in
import os

#external
import biosppy as bp
import numpy as np
from scipy.signal import find_peaks


def get_ecg_analysis(signal_df, pos_label, neg_label, FS):
    """
    Get ecg analysis
    Parameters:
        ecg_signal (array): ecg signal
    Returns:
        dict: dictionary with ecg analysis
    """
    ecg_signal = (signal_df[pos_label].values - signal_df[neg_label].values).reshape(-1)
    ecg_signal = bp.signals.tools.filter_signal(signal=ecg_signal, ftype='FIR', band='bandpass', order=100, frequency=[0.5, 45], sampling_rate=FS)['signal']
    ecg_peaks = bp.signals.ecg.hamilton_segmenter(signal=ecg_signal, sampling_rate=FS)[0]
    ecg_peaks = bp.signals.ecg.correct_rpeaks(signal=ecg_signal, rpeaks=ecg_peaks, sampling_rate=FS, tol=0.05)[0]
    heart_rates = FS * (60/np.diff(ecg_peaks))
    good_hr = np.argwhere((heart_rates>40) & (heart_rates<200)).reshape(-1)
    heart_rates = heart_rates[good_hr]
    hr_info = {'minHR':min(heart_rates), 
               'maxHR':max(heart_rates),
               'avgHR':np.mean(heart_rates),
               'stdHR':np.std(heart_rates),
               'trendHR': np.polyfit(good_hr, heart_rates, 1)[0]}
    return hr_info


def get_resp_analysis(signal_df, pos_label, neg_label, FS):
    """
    Get resp analysis
    Extracts the respiratory signal from the ecg signal
    Calculates the respiratory rate
    Follows the approach from:
        Obrien, C., & Heneghan, C. (2007). A comparison of algorithms for estimation of a respiratory signal from the surface electrocardiogram.
    Parameters:
        ecg_signal (array): ecg signal
    Returns:
        dict: dictionary with resp analysis
    """
    ecg_signal = (signal_df[pos_label].values - signal_df[neg_label].values).reshape(-1)
    ecg_filter = bp.signals.tools.filter_signal(signal=ecg_signal, ftype='FIR', band='highpass', order=20, frequency=0.05, sampling_rate=FS)['signal']
    ecg_peaks = bp.signals.ecg.hamilton_segmenter(signal=ecg_filter, sampling_rate=FS)['rpeaks']
    ecg_peaks = bp.signals.ecg.correct_rpeaks(signal=ecg_filter, rpeaks=ecg_peaks, sampling_rate=FS, tol=0.05)['rpeaks']
    rri = np.diff(ecg_peaks) * 1000 / FS
    mean_rri = int(np.mean(rri))
    # carrier signal
    carrier_signal = np.arange(0, len(ecg_peaks)*mean_rri, mean_rri)*0 + 1
    # the modulated signal is the r peak amplitude spaced at mean rri
    modulated_signal = carrier_signal * ecg_filter[ecg_peaks]
    mod_fs = (1/(mean_rri))*1000
    modulated_signal = bp.signals.tools.filter_signal(signal=modulated_signal, ftype='butter', band='lowpass', order=4, frequency=0.4, sampling_rate=mod_fs)['signal']
    #get edr with linear interpolation
    edr = []
    for idx in range(len(modulated_signal)-1):
        edr_segment = np.interp(x=np.arange(0, rri[idx], 4), xp=[0, rri[idx]], fp=[modulated_signal[idx], modulated_signal[idx+1]])
        edr.append(edr_segment)
    edr_signal = np.hstack(edr)
    # Assuming you have an EDR signal stored in a NumPy array called 'edr_signal'
    # Set a threshold to detect peaks
    threshold, overlap, window_size = 0.5, 0.5, 30  # Time interval for respiratory rate calculation (in seconds)
    # Calculate the number of samples in the window
    window_samples = int(window_size * FS)  # Adjust 'sampling_rate' as per your EDR signal
    # Calculate the number of samples to overlap
    overlap_samples = int(window_samples * overlap)
    # Initialize lists to store the calculated respiratory rates
    respiratory_rates = []
    # Perform sliding window analysis
    start = 0
    while start + window_samples <= len(edr_signal):
        # Extract the EDR signal within the current window
        window = edr_signal[start : start + window_samples]
        # Find peaks in the EDR signal above the threshold
        peaks, _ = find_peaks(window, height=threshold)
        # Calculate the time intervals between consecutive peaks
        peak_intervals = np.diff(peaks)/FS
        # Calculate the average time interval (respiratory cycle duration)
        average_interval = np.mean(peak_intervals)
        # Calculate the respiratory rate in breaths per minute (BPM)
        respiratory_rate = 60 / average_interval
        # Append the respiratory rate to the list
        respiratory_rates.append(respiratory_rate)
        # Move the window
        start += overlap_samples

    resp_info = {'maxRR': np.max(respiratory_rates), 
                    'minRR': np.min(respiratory_rates), 
                    'avgRR': np.mean(respiratory_rates), 
                    'stdRR': np.std(respiratory_rates), 
                    'trendRR': np.polyfit(np.arange(len(respiratory_rates)), respiratory_rates, 1)[0]}
    return resp_info


def cardiac_sqi(segment, sampling_rate=None, threshold=0.9, bit=0):
    """ Eval electrocardiogram quality for one segment with 10s
    """
    LQ, MQ, HQ = 0.0, 0.5, 1.0
    #LQ, MQ, HQ = 3, 2, 1
    
    if bit !=  0:
        if (max(segment) - min(segment)) >= (2**bit - 1):
            return LQ
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if len(segment) < sampling_rate * 5:
        raise IOError('Segment must be 5s long')
    else:
        # TODO: compute ecg quality when in contact with the body
        rpeak1 = bp.signals.ecg.hamilton_segmenter(segment, sampling_rate=sampling_rate)['rpeaks']
        rpeak1 = bp.signals.ecg.correct_rpeaks(signal=segment, rpeaks=rpeak1, sampling_rate=sampling_rate, tol=0.05)['rpeaks']
        if len(rpeak1) < 2:
            return LQ
        else:
            hr = sampling_rate * (60/np.diff(rpeak1))
            quality = MQ if (max(hr) <= 200 and min(hr) >= 40) else LQ
        if quality == MQ:
            templates, _ = bp.signals.ecg.extract_heartbeats(signal=segment, rpeaks=rpeak1, sampling_rate=sampling_rate, before=0.2, after=0.4)
            corr_points = np.corrcoef(templates)
            if np.mean(corr_points) > threshold:
                quality = HQ

    return quality 
