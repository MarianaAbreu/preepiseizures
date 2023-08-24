#
# Get quality analysis for all patients:
#     1) get physiological data 
#     2) calculate quality analysis
#     3) save everything in a csv file per patient: #Hours; %hqECG; %mqECG; %lqECG; SQIresp; SQIacc 
# Created: 24.07.2023
# By: Mariana Abreu
#
# built-in
import json
import os

# external
import numpy as np
import pandas as pd
from mne.io import read_raw_edf


# local
from src import Patient, OpenFileProcessor, quality_utils


def get_acc_signals(datafile, patient, device):
    

    if 'wearable_device_2' in patient.patient_dict.keys(): # has two devices

        if 'EDA' in patient.patient_dict['wearable_sensors']:
            # wristbit is the first device
            if device == 'wristbit':
                sensor_signal = datafile.data_values.iloc[:,:11][['ACCX', 'ACCY', 'ACCZ']]
            else:
                sensor_signal = datafile.data_values.iloc[:,11:][['ACCX', 'ACCY', 'ACCZ']]
        else:
            if device == 'wristbit':
                sensor_signal = datafile.data_values.iloc[:,11:][['ACCX', 'ACCY', 'ACCZ']]
            else:
                sensor_signal = datafile.data_values.iloc[:,:11][['ACCX', 'ACCY', 'ACCZ']]
    else: 
        sensor_signal = datafile.data_values[['ACCX', 'ACCY', 'ACCZ']]
    
    return sensor_signal


def segmentation_and_analysis(signal, type, sampling_rate, window):
    """
    Segmentation and quality analysis
    """
    # segment signal into non-overlapping segments of size window

    if type.lower() == 'ecg':
        window = int(window * sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        threshold = 0.8
        return np.hstack((np.apply_along_axis(quality_utils.cardiac_sqi, 1, segments, sampling_rate, threshold)))
    elif type.lower() == 'pzt':
        window = int(window*sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        return np.hstack((np.apply_along_axis(quality_utils.resp_sqi, 1, segments, sampling_rate)))
    elif type.lower() == 'pzt2':
        window = int(window*sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        return np.hstack((np.apply_along_axis(quality_utils.resp_sqi_2, 1, segments, sampling_rate)))
    
    elif type.lower() == 'eda':
        window = int(window*sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        return np.hstack((np.apply_along_axis(quality_utils.eda_sqi, 1, segments)))
    elif type.lower() == 'bvp':
        window = int(window*sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        return np.hstack((np.apply_along_axis(quality_utils.ppg_sqi, 1, segments, sampling_rate)))
    elif type.lower() == 'acc':
        window = int(window*sampling_rate)
        if len(signal)%window > 0:
            signal = signal[:-(len(signal)%window)] # last bit will be ignored
        segments = signal.reshape(-1, window)
        return np.hstack((np.apply_along_axis(quality_utils.ppg_sqi, 1, segments, sampling_rate)))
    
        
        
    else:
        # TODO
        raise ValueError(f'Unknown quality for {type}, only the ECG quality was computed')


def wearable_quality(patient, sensor, window, device=''):
    """
    Calculate quality for all data from the hospital
    Return dataframe of one quality point per 10 sec
    TODO : Resp and ACC quality
    """
    quality_dir = 'quality'
    datafile = OpenFileProcessor.OpenFileProcessor(mode='EPIBOX')
    fileslist = sorted([file for file in os.listdir(os.path.join(patient.dir, patient.patient_dict['wearable_dir'])) if file.startswith('A20')])
    quality_df = pd.DataFrame()
    status = 'complete'
    quality_file_name = f'{patient.id}_wearable_quality_{sensor}{device}_{status}.parquet'
    # if quality dataframe already exists with status complete -> end processing
    if quality_file_name in os.listdir(quality_dir):
        print('Quality already processed!')
        return pd.read_parquet(os.path.join(quality_dir, quality_file_name))
    else:
        # if quality dataframe already exists with status incomplete -> bring to memory -> else create new df
        quality_file_name = f'{patient.id}_wearable_quality_{sensor}{device}_incomplete.parquet'
        if os.path.isfile(os.path.join(quality_dir, quality_file_name)):
            quality_df = pd.read_parquet(os.path.join(quality_dir, quality_file_name))
        else:
            quality_df = pd.DataFrame(columns=['quality', 'filename', 'datetime'])
    
    error_bool = False
    if not quality_df.empty:
        if quality_df.loc[~(quality_df['filename'] + '.txt').isin(fileslist)].empty:
            return quality_df
    for filename in fileslist:
        filekey = filename.split(os.sep)[-1].split('.')[0]
        if filekey in quality_df['filename'].values:
            print(filekey, ' already evaluated')
            continue
        try:
            datafile.process_chunks(os.path.join(patient.dir, patient.patient_dict['wearable_dir'], filename), patient.patient_dict)
        except Exception as e:
            print(e)
            continue
        if len(datafile.data_values) >= window*1000:
            if sensor != 'acc':
                sensor_signal = datafile.data_values[sensor.upper()[:3]]
            else:
                sensor_signal = get_acc_signals(datafile, patient, device)

            quality_points = segmentation_and_analysis(sensor_signal.values, type=sensor, sampling_rate=datafile.fs, window=window)
        else:
            # print('not enough time to compute quality')
            continue

        df = pd.DataFrame(quality_points, columns=['quality'])
        df['filename'] = filekey
        df['datetime'] = pd.date_range(datafile.start_date, periods=len(quality_points), freq=str(window)+'S')
        quality_df = pd.concat((quality_df, df), ignore_index=True)
    if not error_bool:
        status = 'complete'
    quality_df.to_parquet(os.path.join(quality_dir, quality_file_name), engine='fastparquet')
    return quality_df


def hospital_quality_eeg(patient):
    """
    Calculate quality for all data from the hospital
    Return dataframe of one quality point per 10 sec
    """
    
    filesdir = os.path.join(patient.dir, 'raw_eeg')
    fileslist = sorted([file for file in os.listdir(filesdir) if file.lower().endswith('eeg')])
    mode = 'Hospital'
    datafile = OpenFileProcessor.OpenFileProcessor(mode=mode)
    quality_df = pd.DataFrame()
    status = 'complete'
    quality_file_name = f'{patient.id}_hospital_quality_eeg_{status}.parquet'
    # if quality dataframe already exists with status complete -> end processing
    if quality_file_name in os.listdir(patient.dir):
        print('Quality already processed!')
        return pd.read_parquet(os.path.join(patient.dir, quality_file_name))
    else:
        # if quality dataframe already exists with status incomplete -> bring to memory -> else create new df
        quality_file_name = f'{patient.id}_hospital_quality_incomplete.parquet'
        if os.path.isfile(os.path.join(patient.dir, quality_file_name)):
            quality_df = pd.read_parquet(os.path.join(patient.dir, quality_file_name))
        else:
            quality_df = pd.DataFrame(columns=['quality', 'filename', 'datetime'])
    
    error_bool = False
    for filename in fileslist:
        filekey = filename.split(os.sep)[-1].split('.')[0]
        if filekey in quality_df['filename'].values:
            print(filekey, ' already evaluated')
            continue
        try:
            datafile.process_eeg(os.path.join(patient.dir, 'raw_eeg', filename))
        except Exception as e:
            print(e)
            print('error in file')
            status = 'incomplete'
            continue
        # ecg signal can be inverted, but it does not affect quality - TODO CHECK
        ecg_signal = datafile.data_values.iloc[:, 0] - datafile.data_values.iloc[:, 1]
        quality_points = segmentation_and_analysis(ecg_signal.values, type='ecg', sampling_rate=datafile.fs, window=10)
        
        df = pd.DataFrame(quality_points, columns=['quality'])
        df['filename'] = filekey
        df['datetime'] =pd.date_range(datafile.start_date, periods=len(quality_points), freq='10S')
        quality_df = pd.concat((quality_df, df), ignore_index=True)
    if not error_bool:
        status = 'complete'
    quality_df.to_parquet(quality_file_name, engine='fastparquet')
    return quality_df



def hospital_quality(patient, format='edf'):
    """
    Calculate quality for all data from the hospital
    Return dataframe of one quality point per 10 sec
    """
    quality_dir = 'quality'
    source = patient.patient_dict['hospital_dir']
    filesdir = os.path.join(patient.dir, source)
    fileslist = sorted([file for file in os.listdir(filesdir) if file.lower().endswith(format)])

    mode = 'Hospital'
    datafile = OpenFileProcessor.OpenFileProcessor(mode=mode)
    quality_df = pd.DataFrame()
    status = 'complete'
    quality_file_name = f'{patient.id}_hospital_quality_{format}_{status}.parquet'
    # if quality dataframe already exists with status complete -> end processing
    if quality_file_name in os.listdir(quality_dir):
        print('Quality already processed!')
        return pd.read_parquet(os.path.join(quality_dir, quality_file_name))
    else:
        # if quality dataframe already exists with status incomplete -> bring to memory -> else create new df
        quality_file_name = f'{patient.id}_hospital_quality_incomplete.parquet'
        if os.path.isfile(os.path.join(quality_dir, quality_file_name)):
            quality_df = pd.read_parquet(os.path.join(quality_dir, quality_file_name))
        else:
            quality_df = pd.DataFrame(columns=['quality', 'filename', 'datetime'])
    
    error_bool = False
    for filename in fileslist:
        filekey = filename.split(os.sep)[-1].split('.')[0]
        if filekey in quality_df['filename'].values:
            print(filekey, ' already evaluated')
            continue
        try:
            if format == 'edf':
                datafile.process_edf(os.path.join(patient.dir, patient.patient_dict['hospital_dir'], filename))
            elif format == 'trc':
                datafile.process_trc(os.path.join(patient.dir, patient.patient_dict['hospital_dir'], filename))
            else:
                datafile.process_eeg(os.path.join(patient.dir, 'raw_eeg', filename))
        except Exception as e:
            print(e)
            print('error in file')
            status = 'incomplete'
            continue
        # ecg signal can be inverted, but it does not affect quality - TODO CHECK
        ecg_signal = datafile.data_values.iloc[:, 0] - datafile.data_values.iloc[:, 1]
        quality_points = segmentation_and_analysis(ecg_signal.values, type='ecg', sampling_rate=datafile.fs, window=10)
        
        df = pd.DataFrame(quality_points, columns=['quality'])
        df['filename'] = filekey
        df['datetime'] =pd.date_range(datafile.start_date, periods=len(quality_points), freq='10S')
        quality_df = pd.concat((quality_df, df), ignore_index=True)
    if not error_bool:
        status = 'complete'
    quality_df.to_parquet(os.path.join(quality_dir, quality_file_name), engine='fastparquet')
    if patient == 'RHSO':
        print('deal')
    return quality_df


if __name__ == "__main__":

    # get all patients metadata
    patient_dict = json.load(open('patient_info.json'))
    # select first patient as example
    patient_one = list(patient_dict.keys())[0]
    #patient_list = ['VNVW','RAFI','LDXH','QFRK','YWJN','UIJU','OQQA','RMJL','WOSQ']
    for patient_one in patient_dict.keys():
        if patient_one in ['QDST', 'WMWV', 'PGSE', 'OXDN']:
            continue
        # sufix is HSM or HEM
        print('Processing patient: ', patient_one)
        if patient_dict[patient_one]['source'] == 'HSM':
            folder_dir = f"/Volumes/My Passport/Patients_{patient_dict[patient_one]['source']}"
            if not os.path.isdir(os.path.join(folder_dir, patient_one)):
                print(f'Patient {patient_one} not in folder')
                continue
        else:
            folder_dir = f"/Volumes/T7 Touch/PreEpiSeizures/Patients_{patient_dict[patient_one]['source']}"
        # folder_dir = f'/Users/saraiva/Desktop/Dados'
        
        patient = Patient.Patient(patient_one, folder_dir)

        if patient.dir is None:
            print('Solve')

        if patient.patient_dict['wearable_dir'] != '':
            # GET WEARABLE DATA
            quality_df = wearable_quality(patient, sensor='acc', window=300, device='chestbit')
            if quality_df.empty:
                print('No wearable data')
                continue

        # print(f'{patient_one} hHOSP: {hours_hospital} hWEAR: {hours_wearable} HQhosp: {hq_hospital} HQwear: {hq_hospital} MQhosp: {mq_hospital} MQwear: {mq_wearable} LQhosp: {lq_wearable} LQwear: {lq_wearable}')
        