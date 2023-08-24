# Script to hold quality assessment functions of all signals
# Created by: Mariana Abreu
# Date: 25-07-2023
# Last modified: 14-08-2023
#

# built-in
from datetime import timedelta

# external
import biosppy as bp
import numpy as np
from biosppy.signals.tools import filter_signal, find_extrema
from biosppy.signals import ecg
from scipy.signal import find_peaks, resample
from scipy.stats import pearsonr


def acc_sqi(segment, sampling_rate):
    """
    Calculate the SQI of the accelerometer signal
    The SQI will be the ratio between the frequencies of the normal human motion and the total frequency range (both above the frequency of breathing).
    ---
    Parameters:
        segment: np.array. The segment of the signal to be analyzed
        sampling_rate: int. The sampling rate of the signal
    ---
    Returns:
        sqi: float. The SQI of the signal
    """
    # filter
     
    sig = bp.signals.tools.filter_signal(segment, ftype='butter', band='bandpass',
                                         order=4, frequency=[0.5, 16], sampling_rate=sampling_rate)['signal']
    # divide into 4s segments (since we have 300s this results in 75 segments)
    segments = sig.reshape(75, 4000)
    all_fsqi_acc = [bp.signals.ecg.fSQI(sig, fs=sampling_rate, num_spectrum=[0.8, 5], dem_spectrum=[0.8, 500]) for sig in segments]

    return np.median(all_fsqi_acc)


def respiratory_rate(signal, sampling_rate):
    """
    Calculate the respiratory rate from the EDR or PZT signal
    ---
    Parameters:
        signal: np.array. The signal to be analyzed
        sampling_rate: int. The sampling rate of the signal
    ---
    Returns:
        respiratory_rates: list. The respiratory rates calculated for each window
    """

    # Set a threshold to detect peaks
    threshold, overlap, window_size = 0.5, 0.5, 10  # Time interval for respiratory rate calculation (in seconds)
    # Calculate the number of samples in the window
    window_samples = int(window_size * sampling_rate)  # Adjust 'sampling_rate' as per your EDR signal
    # Calculate the number of samples to overlap
    overlap_samples = int(window_samples * overlap)
    # Initialize lists to store the calculated respiratory rates
    respiratory_rates = []
    # Perform sliding window analysis
    start = 0
    while start + window_samples <= len(signal):
        # Extract the EDR signal within the current window
        window = signal[start : start + window_samples]
        # Find peaks in the EDR signal above the threshold
        peaks, _ = find_peaks(window, height=threshold)
        # Calculate the time intervals between consecutive peaks
        peak_intervals = np.diff(peaks)/sampling_rate
        # Calculate the average time interval (respiratory cycle duration)
        average_interval = np.mean(peak_intervals)
        # Calculate the respiratory rate in breaths per minute (BPM)
        respiratory_rate = 60 / average_interval
        # Append the respiratory rate to the list
        respiratory_rates.append(respiratory_rate)
        # Move the window
        start += overlap_samples

    return respiratory_rates


def resp_sqi_2(segment, sampling_rate):
    """
    Calculate the SQI of the respiratory signal:
    Filter and downsample to 5Hz
    The SQI will be the ratio between the frequencies of 
    a breath rate between 0.1 and 0.5 Hz and the total frequency range.
    ---
    Parameters:
        segment: np.array. The segment of the signal to be analyzed
        sampling_rate: int. The sampling rate of the signal
    ---
    Returns:
        sqi: float. The SQI of the signal
    """

    
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if sum(segment) < 1:
        return 0
    
    
    # BREATH DETECTION
    # 1) Low pass filtered above 1Hz and downsampled to 5Hz
    segment_filt = filter_signal(segment, ftype='butter', band='bandpass', frequency=[0.1, 1.], order=2,
                                sampling_rate=sampling_rate)['signal']
    # downsampling to 5 Hz
    segment_filt_downsampled = resample(segment_filt, int(len(segment_filt)*5 / sampling_rate))
    sqi = ecg.fSQI(segment_filt_downsampled, fs=5, num_spectrum=[0.1,0.5], nseg=len(segment_filt_downsampled))
    return sqi



def resp_sqi(segment, sampling_rate):

    """
    Calculate the SQI of the respiratory signal following the approach of:
    (Charlton et al. 2021) "An impedance pneumography signal quality index: 
    Design, assessment and application to respiratory rate monitoring"
    ---
    Parameters:
        segment: np.array. The segment of the signal to be analyzed
        sampling_rate: int. The sampling rate of the signal
    ---
    Returns:
        sqi: float. The SQI of the signal
    """

    
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if sum(segment) < 1:
        return 0
    
    if len(segment) != sampling_rate * 32:
        raise IOError('Segment must be 32s long')
    
    # BREATH DETECTION
    # divide into 32s segments
    # 1) Low pass filtered above 1Hz and downsampled to 5Hz
    segment_filt = filter_signal(segment, ftype='butter', band='bandpass', frequency=[0.1, 1.], order=2,
                                sampling_rate=sampling_rate)['signal']
    # downsampling to 5 Hz
    segment_filt_downsampled = resample(segment_filt, int(len(segment_filt)*5 / sampling_rate))

    # 2) Normalisation (mean=0, std=1)
    norm_sig = (segment_filt_downsampled - np.mean(segment_filt_downsampled)) / np.std(segment_filt_downsampled)
    # 3) Count Orig algorithm

    # 1) Identify peaks and troughs as local extremas
    peaks = find_extrema(norm_sig, mode='max')['extrema']
    troughs = find_extrema(norm_sig, mode='min')['extrema']
    peaks_amplitudes = norm_sig[peaks]
    troughs_amplitudes = norm_sig[troughs]

    # 2) relevant peaks identified
    peaks_rel = np.array(peaks)[peaks_amplitudes > 0.2 * np.percentile(peaks_amplitudes, 75)]
    # 3) relevant troughs identified
    troughs_rel = np.array(troughs)[troughs_amplitudes < 0.2 * np.percentile(troughs_amplitudes, 25)]

    # 4) 1 peak per consecutive troughs
    try:
        peaks_idx = np.hstack([np.where(((peaks_rel > troughs_rel[i]) * (peaks_rel < troughs_rel[i + 1])))[0] for i in
                            range(len(troughs_rel) - 1)])
    except:
        return 0
    # find peaks rel values of the indexes identified in peaks_idx
    peaks_val = peaks_rel[peaks_idx]
    # breaths will be the times between consecutive peaks found between troughs
    time_breaths = np.diff(peaks_val)
    if len(time_breaths) < 1:
        return 0

    # Evaluate valid breaths
    # 1) std time breaths > 0.25
    quality = True
    if np.std(time_breaths) < 0.25:
        quality = False
        return 1 if quality else 0 # there is signal but with low quality
    # 2) 15% of time breaths > 1.5 or < 0.5 * median breath duration
    bad_breaths = time_breaths[
        ((time_breaths > (1.5 * np.median(time_breaths))) & (time_breaths < (0.5 * np.median(time_breaths))))]
    ratio = len(bad_breaths) / len(time_breaths)
    if ratio >= 0.15:
        quality = False
        return 1 if quality else 0.1 # there is signal but with low quality
    # 3) 60% of segment is occupied by valid breaths
    if (sum(time_breaths) / len(norm_sig)) < 0.6:
        quality = False
        return 1 if quality else 0 # there is signal but with low quality

    # assess similarity of breath morphologies
    # calculate template breath
    # calculate correlation between individual breaths and the template
    # 4) is the mean correlation coeffient > 0.75?
    # get mean breath interval
    mean_breath_interval = np.mean(time_breaths)
    breaths = [norm_sig[int(peaks_val[i] - mean_breath_interval // 2): int(peaks_val[i] + mean_breath_interval // 2)]
            for i in range(len(peaks_val)) if ((peaks_val[i] - mean_breath_interval//2) >= 0)
            and (peaks_val[i] + mean_breath_interval//2 <= len(norm_sig))]

    try:
        mean_template = np.mean(breaths, axis=0)
    except:
        print('ok')

    # compute correlation and mean correlation coefficient
    mean_corr = np.mean([pearsonr(breath, mean_template)[0] for breath in breaths])
    # check if mean corr > 0.75
    if mean_corr < 0.75:
        quality = False

    return 1 if quality else 0 # there is signal but with low quality



from scipy.signal import welch


def racSQI(samples):
    """
    Rate of Amplitude change (RAC)
    It is recomended to be analysed in windows of 2 seconds.
    """
    max_, min_ = max(samples), min(samples)
    amplitude = max_ - min_
    first = max_ if np.argmax(samples) <= np.argmin(samples) else min_
    return abs(amplitude / first)


def eda_sqi(segment):  # -> Timeline
    """
    Suggested by Böttcher et al. Scientific Reports, 2022, for wearable wrist EDA.
    """
    return int(np.mean(segment) > 0.05 and racSQI(segment) < 0.2)

# PPG quality 

def ppg_sqi(segment, sampling_rate=None):  # -> Timeline
    """
    Suggested for wearable wrist PPG by:
        - Glasstetter et al. MDPI Sensors, 21, 2021
        - Böttcher et al. Scientific Reports, 2022
    """

    nperseg = int(4 * sampling_rate)  # 4 s window
    fmin = 0.1  # Hz
    fmax = 5  # Hz

    def spectral_entropy(x, sfreq, nperseg, fmin, fmax):
        if len(x) < nperseg:  # if segment smaller than 4s
            nperseg = len(x)
        noverlap = int(0.9375 * nperseg)  # if nperseg = 4s, then 3.75 s of overlap
        f, psd = welch(x, sfreq, nperseg=nperseg, noverlap=noverlap)
        idx_min = np.argmin(np.abs(f - fmin))
        idx_max = np.argmin(np.abs(f - fmax))
        psd = psd[idx_min:idx_max]
        psd /= np.sum(psd)  # normalize the PSD
        entropy = - np.sum(psd * np.log2(psd))
        N = idx_max - idx_min
        entropy_norm = entropy / np.log2(N)
        return entropy_norm

    return int(spectral_entropy(segment, sampling_rate, nperseg, fmin, fmax) < 0.8)


def cardiac_sqi(segment, sampling_rate=None, threshold=0.9, bit=0):
    """ Eval electrocardiogram quality for one segment with 10s
    3 quality levels: low, medium, high
    low - the signal is not good enough to detect the heart rate
    medium - the signal is good enough to detect the heart rate
    high - the signal is good enough to detect the heart rate and the morphology of the signal is good
    ---
    Parameters
        segment : array. ECG segment with 10s
        sampling_rate : int. Sampling frequency
        threshold : float. Threshold to detect the peaks
        bit : int. Number of bits of the ADC
    ---
    Returns
        quality : int. Quality level. One of HQ, MQ, LQ
    """
    # define the desired quality levels
    LQ, MQ, HQ = 0.0, 0.5, 1.0
    #LQ, MQ, HQ = 3, 2, 1
    
    if bit !=  0:
        # check if the electrodes are disconnected from the body
        if (max(segment) - min(segment)) >= (2**bit - 1):
            return LQ
    if sampling_rate is None:
        raise IOError('Sampling frequency is required')
    if len(segment) < sampling_rate * 5:
        raise IOError('Segment must be 5s long')
    else:
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
