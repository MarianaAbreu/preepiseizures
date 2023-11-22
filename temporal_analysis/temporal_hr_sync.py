# Hospital/ wearable temporal synchronization
# 1) Select several good quality ECG files from the hospital and the wearable (through quality data dataframe)
# 2) See the best match of temporal syncing between the two
# 3) Return the temporal offset between the two
# Created by: Mariana Abreu
# Date: 25th September 2023
#

# built-in´
import json
import os
# external 
import biosppy as bp
import numpy as np
import pandas as pd
from acc_seizure_extraction import find_acc_patients
# local
from preepiseizures.src import Patient, biosignal_processing


def get_data_hr(filedir, source, fs, patient_class, quality_times=None, hosp_class=None):
    """
    Get data from file and return hr
    ----
    Parameters:
        filedir: str - Path to file - whole path and ends with .txt
        source: str - 'hospital' or 'wearable'
        fs: int - Sampling frequency
        quality_times: pd.DataFrame - Dataframe with quality times
        hsm_class: Patient.HSMdata - HSM class
    ----
    Returns:
        ecg_hr: pd.DataFrame - Dataframe with hr data
    """
    
    if source == 'hospital':
        if hosp_class is None:
            raise IOError('Needs the HSM class as input')
        data, start_time = hosp_class.read_file_data(filedir)
        ecg_df = data['ECG2'] - data['ECG1']
        ecg_df.index = pd.date_range(start_time, periods=len(ecg_df), freq=str(1/fs)+'S')
    if source == 'wearable':
        patient_class.datafile.process_chunks(filedir, patient_class.patient_dict)
        start_time = patient_class.datafile.start_date
        data = patient_class.datafile.data_values
        time_range = pd.date_range(start_time, periods=len(data), freq=str(1/fs)+'S')
        data.index = time_range
        data = Patient.correct_patient(data, patient_class.id)
        data.index = data['datetime']  
        ecg_df = data['ECG']  

    if quality_times is not None:
        ecg_df = ecg_df.loc[quality_times.iloc[0]['datetime'] - pd.Timedelta(seconds=30): 
                            quality_times.iloc[-1]['datetime'] + pd.Timedelta(seconds=30)].copy()
        
        pass    
    ecg_hr = biosignal_processing.ecg_processing_to_hr(ecg_df, fs)
    # hr = ecg_df.hr.dropna()
    #if source == 'hospital':
    #    hr_ts = start_time + pd.to_timedelta(hr.index, unit='ms')
    #    hr.index = hr_ts
    return ecg_hr


def get_all_hr_wearable(patient_class, filepath=''):
    """
    Get all hr from wearable files and save to parquet file
    ----
    Parameters:
        filepath: str - Path to parquet file
    ----
    Returns:
        hr_df: pd.DataFrame - Dataframe with all hr data
    """
    if filepath == '':
        filepath = f'data{os.sep}hr{os.sep}{patient_class.id}_wearable_hr.parquet'

    if os.path.isfile(filepath):
        hr_df = pd.read_parquet(filepath, engine='fastparquet')
        return hr_df
    
    pat_quality = patient_class.get_quality_data()

    # else calculate hr 
    all_hr = pd.DataFrame(columns=['hr'])
    i = 0
    for file in pat_quality['filename'].unique():
        print(f"Processing {i}/{len(pat_quality['filename'].unique())}", end='\r')  
        i += 1
        filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'], file + '.txt')
        wear_hr = get_data_hr(os.path.join(patient_class.dir, filedir), source='wearable', fs=int(1000), patient_class=patient_class)
        all_hr = pd.concat((all_hr, wear_hr))

    # i don't know why but column 'hr' is always nan while '0' contains the hr values
    if 'hr' in all_hr.columns:
        all_hr.drop(columns=['hr'], inplace=True)
    all_hr.rename(columns={0: 'hr'}, inplace=True)
    all_hr['timestamp'] = all_hr.index
    all_hr.reset_index(drop=True, inplace=True)
    all_hr.to_parquet(f'data{os.sep}hr{os.sep}{patient_class.id}_wearable_hr.parquet', engine='fastparquet')

    print('Success in creating HR df!')
    return all_hr


def get_all_hr_hospital(patient_class, filepath=''):
    """
    Get all hr from hospital files and save to parquet file
    ----
    Parameters:
        filepath: str - Path to parquet file
    ----
    Returns:
        hr_df: pd.DataFrame - Dataframe with all hr data
    """
    if filepath == '':
        filepath = f'data{os.sep}hr{os.sep}{patient_class.id}_hospital_hr.parquet'
        
    if os.path.isfile(filepath):
        hr_df = pd.read_parquet(filepath, engine='fastparquet')
        return hr_df
    
    hospital_quality = patient_class.get_quality_data(source='hospital')


    if os.path.isdir(os.path.join(patient_class.dir, 'raw_eeg')):
        hosp_dir = os.path.join(patient_class.dir, 'raw_eeg')
    else:
        hosp_dir = os.path.join(patient_class.dir, patient_class.patient_dict['hospital_dir']) 

    all_hr = pd.DataFrame()
    i = 0
    if patient_class.patient_dict['source'] == 'HEM':
        hosp_class = Patient.HEMdata(patient_id=patient_class.id, dir=patient_class.dir)
    else:
        hosp_class = Patient.HSMdata(patient_id=patient_class.id, dir=patient_class.dir)

    if hospital_quality.empty:
        print('No hospital quality data available')
        hospital_files = [file.split('.')[0] for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf', 'trc']]
    else:
        hospital_files = hospital_quality['filename'].unique()
    hospital_files_ending = [file for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf', 'trc']]
    for filename in hospital_files:
        file_with_ending = [file for file in hospital_files_ending if filename in file][0]
        print(f"Processing {i}/{len(hospital_files)}", end='\r')  
        i += 1
        filedir = os.path.join(hosp_dir, file_with_ending)
        hosp_hr = get_data_hr(filedir, source='hospital', fs=int(hosp_class.FS), patient_class=patient_class, hosp_class=hosp_class)
        all_hr = pd.concat([all_hr, hosp_hr])
    if 'hr' in all_hr.columns:
        all_hr.drop(columns=['hr'], inplace=True)
    all_hr.rename(columns={0: 'hr'}, inplace=True)
    all_hr['timestamp'] = all_hr.index
    all_hr.reset_index(drop=True, inplace=True)
    all_hr.to_parquet(f'data{os.sep}hr{os.sep}{patient_class.id}_hospital_hr.parquet', engine='fastparquet')
    print('Success in creating HR df!')

    return all_hr


def process_hr(hr_df):
    """
    Receives an hr dataframe with timestamps column and hr column.
    Resamples and processes the HR signal to have a clean and continuous signal.
    ----
    Parameters:
        hr_df: pd.DataFrame - Dataframe with hr data
    ----
    Returns:
        hr_df: pd.DataFrame - Dataframe with processed hr data
    """
    hr_df = hr_df.set_index('timestamp')
    # median filter

    hr_df = hr_df.resample('5S').median()
    hr_df = hr_df.interpolate(method='linear', limit_direction='both')
    hr_df = hr_df.dropna()
    # clean even more
    # hr_df = hr_df.resample('30S').median()
    hr_filt = bp.signals.tools.filter_signal(signal=hr_df['hr'].values, ftype='butter', band='lowpass', order=4, frequency=0.0005, sampling_rate=0.2)['signal']
    hr_df['hr'] = hr_filt
    return hr_df


def compute_correlation(df_h, df_w, sampling_rate=0.2):
    """
    Computes the correlation between two hr signals
    ----
    Parameters:
        df_h: pd.DataFrame - Dataframe with hr data from hospital
        df_w: pd.DataFrame - Dataframe with hr data from wearable
    ----
    Returns:
        df_all: pd.DataFrame - Dataframe with correct timestamps and hr data from both sources
    """
    df_h = df_h.copy()
    df_w = df_w.copy()
    synchronize_info = bp.signals.tools.synchronize(x=df_h['hr'].values, y=df_w['hr'].values)
    delay = synchronize_info['delay']
    print(delay)
    # reset index to hospital timestamps with delay
    df_w.index = df_h.index[:len(df_w)] + pd.Timedelta(seconds=delay * (1/sampling_rate))
    # rename hr columns to differentiate between hospital and wearable
    df_w.rename(columns={'hr': 'hr_w'}, inplace=True)
    df_h.rename(columns={'hr': 'hr_h'}, inplace=True)
    # join all dataframes
    df_all = pd.concat([df_h, df_w], axis=1)
    return df_all, df_w


if __name__ == '__main__':

    patient_dict = json.load(open('patient_info.json', 'r'))
    patient_list = patient_dict.keys()
    for patient in patient_list:
        
        patient_class = Patient.patient_class(patient)
        if patient_class is None:
            continue
        ## GET HR DATA
        hr_wear = get_all_hr_wearable(patient_class)
        hr_hosp = get_all_hr_hospital(patient_class)
        if hr_wear.empty or hr_hosp.empty:
            print('No HR data available')
            continue
        hr_new = process_hr(hr_wear)
        hr_new_2 = process_hr(hr_hosp)
        df_all, df_w = compute_correlation(hr_new_2, hr_new)
        temporal_shift = df_all['hr_w'].dropna().index[0] - hr_new.index[0]
        print('temporal shift: ', temporal_shift)
        patient_info = json.load(open(os.path.join('patient_info.json'), 'r'))
        patient_info[patient]['temporal_shift'] = str(temporal_shift)
        json.dump(patient_info, open('patient_info.json', 'w'), indent=4)
        pass

    
    
    
    #pat_quality = patient_class.get_quality_data()



    # good_quality = pat_quality.loc[pat_quality['quality'] > 0]
    # good_files = good_quality['filename'].unique()
    # good_times = good_quality.loc[good_quality['filename'].isin(good_files)]
    # #
    # # HOSPITAL DATA ----
    # hospital_quality = patient_class.get_quality_data(source='hospital')
    # hospital_quality_good = hospital_quality.loc[hospital_quality['quality'] > 0]
    # hospital_data = pd.DataFrame()
    # hsm_class = Patient.HSMdata(patient_id=patient_class.id, dir=patient_class.dir)
    # if os.path.isdir(os.path.join(patient_class.dir, 'raw_eeg')):
    #     hosp_dir = os.path.join(patient_class.dir, 'raw_eeg')
    # else:
    #     hosp_dir = os.path.join(patient_class.dir, patient_class.patient_dict['hospital_dir'])    
    # hospital_files = [file for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf']]

    # # # Run wearable files and get hr
    # # all_hr = pd.DataFrame(columns=['hr'])
    # # i = 0
    # # for file in pat_quality['filename'].unique():
    # #     print(f"Processing {i}/{len(pat_quality['filename'].unique())}", end='\r')  
    # #     i += 1
    # #     filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'], file + '.txt')
    # #     wear_hr = get_data(os.path.join(patient_class.dir, filedir), source='wearable', fs=int(1000))
    # #     all_hr = pd.concat((all_hr, wear_hr))
    # # # i don't know why but column 'hr' is always nan while '0' contains the hr values
    # # if 'hr' in all_hr.columns:
    # #     all_hr.drop(columns=['hr'], inplace=True)
    # # all_hr.rename(columns={0: 'hr'}, inplace=True)
    # # all_hr['timestamp'] = all_hr.index
    # # all_hr.reset_index(drop=True, inplace=True)
    # # all_hr.to_parquet(f'{patient}_wearable_hr.parquet', engine='fastparquet')
        
    # print('Success!')
    # all_hr = pd.DataFrame()
    # i = 0
    # for filename in hospital_quality['filename'].unique():
    #     file_with_ending = [file for file in hospital_files if filename in file][0]

    #     print(f"Processing {i}/{len(hospital_quality['filename'].unique())}", end='\r')  
    #     i += 1
    #     filedir = os.path.join(hosp_dir, file_with_ending)
    #     hosp_hr = get_data(filedir, source='hospital', fs=int(hsm_class.FS))
    #     all_hr = pd.concat([all_hr, hosp_hr])
    # if 'hr' in all_hr.columns:
    #     all_hr.drop(columns=['hr'], inplace=True)
    # all_hr.rename(columns={0: 'hr'}, inplace=True)
    # all_hr['timestamp'] = all_hr.index
    # all_hr.reset_index(drop=True, inplace=True)
    # all_hr.to_parquet(f'{patient}_hospital_hr.parquet', engine='fastparquet')
        
    # print('Success!')
    # # for file in good_files:
    # #     times_file = good_quality.loc[good_quality['filename']==file]
    # #     if len(times_file) < 10:
    # #         print('Only one good quality row for file...')
    # #     else:
    # #         hospital_times = hospital_quality.loc[hospital_quality['datetime'].between(times_file.iloc[0]['datetime'] - pd.Timedelta(hours=1), 
    # #                                                                                    times_file.iloc[-1]['datetime'] + pd.Timedelta(hours=1))]
    # #         hospital_good_files = hospital_times['filename'].unique()
    # #         hosp_hr = pd.DataFrame()
    # #         filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'], file + '.txt')
    # #         wear_hr = get_data(filedir, source='wearable', fs=int(1000), quality_times=times_file)
    # #         for filename in hospital_good_files:
                
    # #             file_with_ending = [file for file in hospital_files if filename in file][0]
    # #             filedir = os.path.join(hosp_dir, file_with_ending)
    # #             hosp_hr_temp = get_data(filedir, source='hospital', fs=int(hsm_class.FS))
    # #             hosp_hr = pd.concat([hosp_hr, hosp_hr_temp])
            
    # #             #sync = bp.signals.tools.synchronize(x=hr.values, y=)
                
    # #         pass