# Biosignal Processing Script
# ACC processing
# created by: Mariana Abreu
# date: 29 August 2023

# built-in


# external
import biosppy as bp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample
from scipy.spatial import KDTree
from statsmodels.tsa.seasonal import seasonal_decompose


# local

def acc_processing(acc_df, sampling_rate):
    """
    Receives ACC tri axis channels in a dataframe format with a column of datetime
    Returns a new dataframe with the same temporal index with additional columns such as magnitude and activity index and filtered data
    """
    # get acc columns
    acc_cols = [col for col in acc_df.columns if 'ACC' in col]
    acc_data = acc_df[acc_cols]
    # filter_data
    for col in acc_cols:
        acc_data[col] = bp.signals.tools.filter_signal(signal=acc_data[col].values,ftype='butter', band='bandpass', order=4, frequency=[1, 25], sampling_rate=sampling_rate)['signal']
    # acc magnitude
    acc_data.loc[:, 'ACCM'] = (acc_data**2).sum(axis=1)**0.5
    # activity index is the sum of 12 standard deviations computed for 5s periods without overlap
    # window size for activity index
    n = sampling_rate * 5 
    acc_index = acc_data['ACCM'].rolling(n, step=n).std().rolling(12).sum()
    acc_data.loc[:, 'ACCI'] = acc_index
    return acc_data

def ecg_processing_segment(ecg, sampling_rate=1000):
    """
    Receives ECG single channel in a dataframe format with a column of datetime
    Returns a new dataframe with heart rate 
    """
    #try:
    #    ecg.max() # if apply is used
    #except:
    ecg = ecg[1]['filtered'] # if map is used
    #if (ecg.max() - ecg.min() > 1000 or len(ecg.dropna())<1) :
    #    return None
    if len(ecg) < sampling_rate:
        return pd.Series(np.nan, index=[ecg.index[0]], name='hr')
    if (ecg.kurt() < 3 or (ecg.var() < 400 and ecg.kurt()>30)):
        return pd.Series(np.nan, index=[ecg.index[0]], name='hr')
    ecg_peaks = bp.signals.ecg.hamilton_segmenter(signal=ecg, sampling_rate=sampling_rate)['rpeaks']
    ecg_peaks = bp.signals.ecg.correct_rpeaks(signal=ecg, rpeaks=ecg_peaks, sampling_rate=sampling_rate, tol=0.05)['rpeaks']
    # heart rate
    # print(ecg.kurt(), ecg.var())
    ecg_hr_times = ecg.iloc[ecg_peaks[1:]].index
    ecg_hr = 60/(np.diff(ecg_peaks) / sampling_rate)
    # replace by nans all values outside the proper range
    mask = (ecg_hr < 40) | (ecg_hr > 150)
    ecg_hr[mask] = np.nan
    
    return pd.Series(ecg_hr, index=ecg_hr_times, name='hr')

def ecg_processing_to_hr(ecg_df, sampling_rate):
    """
    Receives ECG single channel in a dataframe format with a column of datetime
    Returns the same dataframe with additional columns such as heart rate and filtered data
    """
    ecg_new = pd.DataFrame(index=ecg_df.index, columns=['filtered'])
    # get ecg filtered
    ecg = bp.signals.tools.filter_signal(signal=ecg_df.values,ftype='FIR', band='bandpass', order=200, frequency=[1, 40], sampling_rate=sampling_rate)['signal']
    # acc magnitude
    ecg_new.loc[:,'filtered'] = ecg
    segments = ecg_new.resample(rule='20S')
    # hr_segments = segments.apply(ecg_processing_segment)
    hr_segments = pd.concat(list(map(ecg_processing_segment, segments)))
    
    #hr_ts = ecg_df.iloc[ecg_peaks[1:]].index
    #ecg_new.loc[hr_ts, 'hr'] = ecg_hr
    # join to original table
    #return ecg_new
    return hr_segments


def ecg_segment_norm(ecg_df):
    """
    Receives ECG single segment in a dataframe with datetime in index
    Returns the same dataframe with normalised data after checking quality
    """
    # bad quality
    if ecg_df.empty:
        return None 
    
    i = ecg_df.index[0]
    ecg_df.index = ecg_df['timestamp']
    ecg_df.drop(columns=['timestamp'], inplace=True)
    ecg_new = ecg_df.copy()
    if (ecg_df['ECG'].kurt() < 3 or (ecg_df['ECG'].var() < 400 and ecg_df['ECG'].kurt() > 30)):
        return None 
    ecg_new['ECG'] = (ecg_df - ecg_df.min()) / (ecg_df.max() - ecg_df.min())
    ecg_new.loc[:, 'id'] = i
    return ecg_new


def ecg_processing(ecg_df, sampling_rate):
    """
    Receives ECG single channel in a dataframe format with a column of datetime
    Returns the same dataframe with filtered data and resampled to 80Hz
    """
    # get ecg filtered
    ecg = bp.signals.tools.filter_signal(signal=ecg_df.values,ftype='FIR', band='bandpass', order=200, frequency=[1, 40], sampling_rate=sampling_rate)['signal']
    # resampling to 80Hz
    resampled = resample(ecg, int(len(ecg)*80 / sampling_rate))
    resampled_time = pd.date_range(ecg_df.index[0], ecg_df.index[-1], periods=len(resampled))
    ecg_df = pd.DataFrame(resampled, index=resampled_time, columns=['ECG'])    
    # check inversion: calculate rpeaks for the original and inverted signal
    good_ecg = None
    start_time = ecg_df.index[0] + pd.Timedelta(seconds=300)
    while good_ecg is None:
        temp_ecg = ecg_df.loc[start_time:start_time + pd.Timedelta(seconds=10)]
        if (temp_ecg['ECG'].kurt()) < 5 or (temp_ecg['ECG'].kurt()) > 100:
            start_time += pd.Timedelta(seconds=120)
        else:
            good_ecg = temp_ecg.copy()
        if start_time >= ecg_df.index[-1]:
            print('problem')

    if (good_ecg is None) or (len(good_ecg) < 400):
        good_ecg = ecg_df.copy()

    r_peaks = bp.signals.ecg.hamilton_segmenter(signal=good_ecg['ECG'], sampling_rate=80)['rpeaks']
    r_peaks = bp.signals.ecg.correct_rpeaks(signal=good_ecg['ECG'], rpeaks=r_peaks, sampling_rate=80, tol=0.05)['rpeaks']
    r_peaks_i = bp.signals.ecg.hamilton_segmenter(signal=good_ecg['ECG']*-1, sampling_rate=80)['rpeaks']
    r_peaks_i = bp.signals.ecg.correct_rpeaks(signal=good_ecg['ECG']*-1, rpeaks=r_peaks_i, sampling_rate=80, tol=0.05)['rpeaks']
    # if the difference between the two is less than 5 in absolute, the calculation is correct
    # Define a threshold for how close numbers need to be
    threshold = 5
    # Create new lists with only the elements that are close to an element in the other list
    #rp_new = np.array([i for i in r_peaks if any(abs(i - j) <= threshold for j in r_peaks_i)])
    #rpi_new = np.array([i for i in r_peaks_i if any(abs(i - j) <= threshold for j in r_peaks)])
    tree = KDTree(r_peaks_i.reshape(-1, 1))

    # Query the tree for the nearest neighbor of each point in 'x' within the threshold
    distances, indices = tree.query(r_peaks.reshape(-1, 1), distance_upper_bound=threshold)

    # Filter 'x' and 'y' based on the query results
    rp_new = r_peaks[distances != np.inf]
    rpi_new = r_peaks_i[indices[distances != np.inf]]
    if (len(rp_new) != len(rpi_new)):
        print('Problem with the first 10 minutes of the signal.')
        # delete first or last peak
        
    r_peak_diff = np.median(rp_new - rpi_new)
    if abs(r_peak_diff) < 5:
        if r_peak_diff > 0: # signal is inverted if the difference is positive
            ecg_df['ECG'] = ecg_df['ECG']*-1
            print('ECG signal inverted.')
    else:
        print('ECG signal not inverted. Problem with the first 10 minutes of the signal.')
    return ecg_df


def resp_rate(signal, sampling_rate):
    """
    Get respiratory rate from a respiratory signal
    ----
    Parameters:
        signal (array): respiratory signal
        sampling_rate (int): sampling rate
    Returns:
        respiratory rate (float)
    """

    threshold, overlap, window_size = 0.5, 0.5, 30  # Time interval for respiratory rate calculation (in seconds)
    # Calculate the number of samples in the window
    window_samples = int(window_size * sampling_rate)  # Adjust 'sampling_rate' as per your EDR signal
    # Calculate the number of samples to overlap
    overlap_samples = int(window_samples * overlap)
    # Initialize lists to store the calculated respiratory rates
    respiratory_rates = []
    # Perform sliding window analysis
    start = 0
    respiratory_times = []
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
        #if respiratory_rate == 2400.0:
        #    print('problem')
        # Append the respiratory rate to the list
        respiratory_rates.append(respiratory_rate)
        # Move the window
        respiratory_times.append(start + window_samples)
        start += overlap_samples
    return respiratory_rates, respiratory_times


def resp_processing(resp_df, sampling_rate):
    """
    Receives PZT channels in a dataframe format with a column of datetime
    Returns the same dataframe with additional columns such as filtered data and respiratory rate

    """
    # get acc columns
    # window size

    # acc magnitude
    # 1) Low pass filtered above 1Hz and downsampled to 5Hz
    # if int(resp_df.std()) == 0:
    #     resp_df['resp_rate'] = [0] * len(resp_df)
    #     return resp_df
    # segment_filt = bp.signals.tools.filter_signal(resp_df.values, ftype='butter', band='bandpass', frequency=[0.1, 1.], order=2,
    #                                               sampling_rate=sampling_rate)['signal']
    # # downsampling to 5 Hz
    # segment_filt_downsampled = resample(segment_filt, int(len(segment_filt)*5 / sampling_rate))

    # # 2) Normalisation (mean=0, std=1)
    # norm_sig = (segment_filt_downsampled - np.mean(segment_filt_downsampled)) / np.std(segment_filt_downsampled)
    # resp_rates = resp_rate(norm_sig, sampling_rate=5)

    # filtering piezoelectric respiratory signal same processing as described in the morphological autoencoder paper
    if len(resp_df) < sampling_rate*20:
        return pd.DataFrame(columns=['RESP', 'ECG'])
    resp = bp.signals.tools.filter_signal(resp_df['PZT'].values, ftype='butter', band='bandpass', frequency=[0.01, 0.35], order=2,
                                                  sampling_rate=sampling_rate)['signal']
    resampled = resample(resp, int(len(resp)*80 / sampling_rate))
    resampled_time = pd.date_range(resp_df.index[0], resp_df.index[-1], periods=len(resampled))
    # ecg filter
    ecg = bp.signals.tools.filter_signal(signal=resp_df['ECG'].values,ftype='FIR', band='bandpass', order=200, frequency=[1, 40], sampling_rate=sampling_rate)['signal']
    # resampling to 80Hz
    resampled_ecg = resample(ecg, int(len(ecg)*80 / sampling_rate))
    new_resp_df = pd.DataFrame(resampled, index=resampled_time, columns=['RESP'])   
    new_resp_df['ECG'] = resampled_ecg
    return new_resp_df
    