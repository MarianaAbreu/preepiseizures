#
# Compute overall quality metrics following the work of:
# 'Bottcher, S. et al. (2022). Data quality evaluation in wearable monitoring'
# 
# Created: 02.08.2023
# By: Mariana Abreu
#
# built-in
import json
import os
from datetime import datetime

# external
import numpy as np
import pandas as pd


# local
from src import Patient, OpenFileProcessor, quality_utils


def correct_patient(df, patient, patient_dict=json.load(open('patient_info.json'))):
    """ 
    AGGA patient has one file that does not belong to the same date as the others.
    This function removes the rows related to that file.
    ---
    Parameters:
        df: pd.DataFrame
            The dataframe with the quality data of the patient
    ---
    Returns:
        df: pd.DataFrame
            The dataframe with the quality data of the patient without the rows related to the wrong file
    """
    dir_ = f"/Volumes/My Passport/Patients_HSM/{patient}/{patient_dict[patient]['wearable_dir']}/annotations.txt"
    if os.path.exists(dir_):
        annotations = pd.read_csv(dir_, header=None, sep='  ')
        # true timestamp 
        true_time = datetime.strptime(annotations.iloc[0][4], '%Y-%m-%d %H:%M:%S.%f')
        # corresponding timestamp saved in the Bitalino files
        unsure_time = datetime.strptime(annotations.iloc[0][6], '%Y-%m-%d %H:%M:%S.%f')
        # correct the timestamps based on the lag between the two timestamps
        df['datetime'] += (true_time - unsure_time)
        return df
    elif patient == 'QFRK':
        start_date = datetime(2019,7,7,10,4,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'OQQA':
        start_date = datetime(2019,12,10,10,20,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'UIJU':
        start_date = datetime(2019,7,30,10,20,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'OXDN':
        timestamp_wrong = df.where(df['datetime'].diff() > pd.Timedelta(days=1)).dropna().index[0]
        return df.iloc[timestamp_wrong:]
    elif patient == 'AGGA':
        timestamp_wrong = df.where(df['datetime'].diff() > pd.Timedelta(days=1)).dropna().index[0]
        return df.iloc[:timestamp_wrong]
    elif patient == 'VNVW':
        timestamp_wrong = df.where(df['datetime'].diff() > pd.Timedelta(days=1)).dropna().index[0]
        lag = df.iloc[timestamp_wrong]['datetime'].date()-df.iloc[timestamp_wrong-1]['datetime'].date()
        df.loc[:timestamp_wrong,'datetime'] = df.iloc[:timestamp_wrong]['datetime'] + lag
        return df
    else:
        return df


def duration_metrics(df, sensor=''):
    """
    Calculate the overall duration of the data
    ---
    Parameters:
        df: pd.DataFrame
            The dataframe with the quality data of the patient
    ---
    Returns:
        dict
            A dictionary of duration metrics: data completeness, total duration, recorded duration
    """
    if sensor == 'pzt':
        window_size = 32
    else:
        window_size = 10

    if len(df.loc[df['datetime'].diff() < pd.Timedelta(seconds=window_size)]) > 0:
        repeated_lines = df.loc[df['datetime'].diff() < pd.Timedelta(seconds=window_size)][::2].index
        df.drop(repeated_lines, inplace=True)
    
    expected_duration = df['datetime'].iloc[-1] - df['datetime'].iloc[0]
    recorded_duration = (len(df)-1) * pd.Timedelta(seconds=window_size)
    data_completeness = recorded_duration / expected_duration
    return {'data_completeness': data_completeness,
            'total_duration': expected_duration, 
            'recorded_duration': recorded_duration}


def quality_metrics_ecg(df):
    """
    Calculate the overall quality metrics of the ECG data
    ---
    Parameters:
        df: pd.DataFrame. The dataframe with the quality data of the patient
    ---
    Returns:
        dict. A dictionary of quality metrics: high quality, medium quality, low quality
    """
    if df.loc[df['quality'] != 0].empty:
        ecg_high_quality = 0
        ecg_medium_quality = 0
        ecg_onbody = 0
    else:
        ecg_high_quality = len(df.loc[df['quality']==1])/len(df)
        ecg_medium_quality = len(df.loc[df['quality']==0.5])/len(df)
        ecg_onbody = len(df.loc[df['quality']!=0])/len(df)

    return {'ecg_hq': ecg_high_quality,
            'ecg_mq': ecg_medium_quality, 
            'onbody': ecg_onbody}


def quality_metrics_binary(df, sensor=''):
    """
    Calculate the overall quality metrics of the ECG data
    ---
    Parameters:
        df: pd.DataFrame. The dataframe with the quality data of the patient
    ---
    Returns:
        dict. A dictionary of quality metrics: high quality, medium quality, low quality
    """
    if df.loc[df['quality'] != 0].empty:
        sensor_quality = 0
    else:
        sensor_quality = len(df.loc[df['quality']==1])/len(df)

    return {sensor: sensor_quality}


def get_quality_df(patient, source='wearable', sensor=''):
    """
    Get the quality dataframe of one patient
    ---
    Parameters:
        patient: str. The patient ID
        source: str. The source of the data. Either 'wearable' or 'hospital'
    ---
    Returns:
        df: pd.DataFrame. A dataframe with the quality data of one patient
    """
    if source == 'hospital':
        df_dir = [file for file in os.listdir('quality') if f'{patient}_{source}' in file]
    elif source == 'wearable':
        df_dir = [file for file in os.listdir('quality') if f'{patient}_{source}_quality_{sensor}' in file]
    
    if len(df_dir) > 1:
        print('TODO: more than one wearable file')
    elif len(df_dir) == 1:
        df = pd.read_parquet(f'quality/{df_dir[0]}')
    else:
        # print(f'No quality data from {source} for patient {patient}')
        return None

    if df.empty:
        return None
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(inplace=True)
    df = correct_patient(df, patient)
    return df



def quality_single_patient(patient, source='wearable', sensor='', device=''):
    """
    Calculate the overall quality metrics of the wearable data of one patient
    ---
    Parameters:
        patient: str. The patient ID
        source: str. The source of the data. Either 'wearable' or 'hospital'
    ---
    Returns:
        df_all: pd.DataFrame. A dataframe with the overall quality metrics of the wearable data of one patient"""

    if source == 'hospital':
        df_dir = [file for file in os.listdir('quality') if f'{patient}_{source}' in file]
    elif source == 'wearable':
        df_dir = [file for file in os.listdir('quality') if f'{patient}_{source}_quality_{sensor}{device}' in file]
    
    if len(df_dir) > 1:
        print('TODO: more than one wearable file')
    elif len(df_dir) == 1:
        df = pd.read_parquet(f'quality/{df_dir[0]}')
    else:
        # print(f'No quality data from {source} for patient {patient}')
        return None

    if df.empty:
        return None
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(inplace=True)
    if (patient in ['OXDN', 'AGGA', 'VNVW'] and source=='wearable'):
        df = correct_patient(df, patient)
    # df.sort_values(by='index', inplace=True)
    df_duration = duration_metrics(df, sensor)
    if sensor == 'ecg':
        df_sensor = quality_metrics_ecg(df)
    else:
        df_sensor = quality_metrics_binary(df, sensor)

    
    df_all = pd.DataFrame({'patient':patient, **df_duration, **df_sensor}, index=[0])

    if (df_duration['data_completeness'] > 1.1 or df_duration['data_completeness'] < 0):
        print(f'Patient {patient} has data completeness {df_duration["data_completeness"]}')
    return df_all


def wearable_quality(sensor, device=''):
    """
    Calculate the overall quality metrics of the wearable data of all patients
    ---
    Returns:
        df_all: pd.DataFrame. A dataframe with the overall quality metrics of the wearable data of all patients
    """

    wearable_patients = [file.split('_')[0] for file in os.listdir('quality') if 'wearable' in file]
    
    df_all = pd.DataFrame()
    for patient in wearable_patients:
        df_patient = quality_single_patient(patient, source='wearable', sensor=sensor, device='')
        if df_patient is not None:
            df_all = pd.concat([df_all, df_patient], ignore_index=True)
    return df_all


def hospital_quality():
    """
    Calculate the overall quality metrics of the hospital data of all patients
    ---
    Returns:
        df_all: pd.DataFrame. A dataframe with the overall quality metrics of the hospital data of all patients
    """
    hospital_patients = [file.split('_')[0] for file in os.listdir('quality') if 'hospital' in file]
    
    df_all = pd.DataFrame()
    for patient in hospital_patients:
        df_patient = quality_single_patient(patient, 'hospital', sensor='ecg')
        if df_patient is not None:
            df_all = pd.concat([df_all, df_patient], ignore_index=True)
    return df_all


def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('hour',        60*60),
        ('minute',      60),
        ('second',      1)
    ]

    strings=[]
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            strings.append(str(period_value))

    return ":".join(strings)


def final_table(source, patient_list=[], device='', sensor='ecg'):

    if sensor == 'ecg':
        cols = ['data_completeness', 'onbody', 'ecg_hq', 'ecg_mq', 'total_duration']
    else: 
        cols = ['data_completeness', sensor, 'total_duration']

    if source == 'hospital':
        df = hospital_quality()
    elif source == 'wearable':
        df = wearable_quality(sensor = sensor, device=device).drop_duplicates()
    
    if patient_list == []:
        if device == 'armbit':
            armbit_patients = ['QDST', 'WMWV', 'PGSE', 'OXDN']
            df = df.loc[df['patient'].isin(armbit_patients)].drop_duplicates()
        elif device == 'wristbit':
            wristbit_patients = ['WZFI','VNVW','RAFI','LDXH','QFRK','YWJN','UIJU','XEUD','OQQA','RMJL','WOSQ']
            df = df.loc[df['patient'].isin(wristbit_patients)].drop_duplicates()
        elif device == 'chestbit':
            unwanted_patients = ['QDST', 'WMWV', 'PGSE', 'OXDN', 'WZFI']
            df = df.loc[~df['patient'].isin(unwanted_patients)].drop_duplicates()
    else:
        df = df.loc[df['patient'].isin(patient_list)]
    df_mean = df[cols].mean()
    df_mean['total_duration'] = td_format(df_mean['total_duration'])
    df_median = df[cols].median()
    df_median['total_duration'] = td_format(df_median['total_duration'])
    df_std = df[cols].std()
    df_std['total_duration'] = td_format(df_std['total_duration'])
    df_min = df[cols].min()
    df_min['total_duration'] = td_format(df_min['total_duration'])
    df_max = df[cols].max()
    df_max['total_duration'] = td_format(df_max['total_duration'])
    
    table = pd.concat([df_mean, df_median, df_std, df_min, df_max], axis=1).T 
    table.index = ['mean', 'median', 'std', 'min', 'max']
    return table, len(df), td_format(df['total_duration'].sum()), df['total_duration'].sum()
    

if __name__ == '__main__':

    # load quality of one patient
    patient_dict = json.load(open('patient_info.json'))
    for sensor in ['ecg']:
        table_w, wN, wdur, ll = final_table(source='wearable', device='chestbit', sensor=sensor)
        print(f'{sensor.upper()} \n {table_w} \n N patients: {wN} \n Total duration: {wdur}')
    print('ok')

