# Script to run HRV features for a single patient
# Besides the features, 

# Import libraries

# built-in libraries
import os

# third-party libraries
import biosppy as bp
import biosppy.quality as quality
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# local libraries
from preepiseizures.src import Patient, Biosignal, Prediction, biosignal_processing

import warnings
warnings.filterwarnings("ignore")


def fetch_original_data(patient_info):
    """
    """
    sensor = 'ecg'
    source = 'hospital'
    device = 'chestbit'
    biosignal = Biosignal.Biosignal(sensor, device, source)
    check = biosignal.get_ecg_hospital_data(patient_class=patient_info)

        
    if check == 1:
        print('ECG data loaded.')
    else:
        print('ECG data not loaded.')
        return pd.DataFrame()

    # get data around seizure
    all_data = pd.concat([pd.read_parquet(os.path.join('data', 'segments', patient_info.id, filename)) 
                          for filename in sorted(os.listdir(f'data{os.sep}segments{os.sep}{patient_info.id}'))])
    all_data['datetime'] = all_data.index
    all_data.set_index('datetime', inplace=True)

    return all_data


def fetch_rpeaks(data):
    """
    """


    data['1H'] = data.index
    data['1H'] = data['1H'].dt.round('1H')
    rpeaks = data.groupby(by='1H').apply(lambda x: biosignal_processing.get_rpeaks(x))
    for i in range(len(rpeaks)):
        data.loc[data.loc[data['1H'] == rpeaks.index[i]].iloc[rpeaks.iloc[i]].index, 'rpeaks'] = 1

    return data


def fetch_quality(data):
    """
    """

    data['10S'] = data.index
    data['10S'] = data['10S'].dt.round('10S')
    quality_points = data.groupby(by='10S').apply(lambda x: biosignal_processing.get_ecg_quality(x))
    data_final = data.merge(quality_points.rename('quality'), left_on='10S', right_index=True)

    return data_final


def fetch_data_prepared(patient_data, patient_info, quality=True, rpeaks=True, save=True):
    """Get all data from patient, with quality and rpeaks if desired.

    Parameters:
        dataname (str): datapath of the patient
        quality (bool, optional): if quality is to be calculated. Defaults to True.
        rpeaks (bool, optional): if rpeaks are to be calculated. Defaults to True.
        save (bool, optional): if data is to be saved. Defaults to True.
    """

    if os.path.exists(patient_data):
        data = pd.read_pickle(patient_data)
    else:
       data = fetch_original_data(patient_info)
       if data.empty:
           return pd.DataFrame()
    
    if quality:
        if 'quality' not in data.columns:
            print('Processing quality...')
            data = fetch_quality(data)
    if rpeaks:
        if 'rpeaks' not in data.columns:
            print('Processing rpeaks...')
            data = fetch_rpeaks(data)
    if save:
        data.to_pickle(patient_data)

    return data



if __name__ == '__main__':

    patient = 'SYRH'
    patient_info = Patient.patient_class(patient)
    patient_info.get_seizure_annotations()
    patient_info.seizure_table

    patient_data = os.path.join('data', patient + '_data_all.pkl')

    data = fetch_data_prepared(patient_data, quality=True, rpeaks=True, save=True)
    hrv_data = biosignal_processing.get_hrv_train(data)
    hrv_data.to_pickle(os.path.join('data', patient + '_hrv_data.pkl'))