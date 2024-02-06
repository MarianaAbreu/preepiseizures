# Script to Train Autoencoder for Respiratory Modulation in Data from the ChestBIT
#
# Author: Mariana Abreu
# Date: 2024-01-25
#

# built-in libraries
import os
import pickle

# third-party libraries
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# local libraries
from preepiseizures.src import Patient
import Respiration_2023


def get_patient_data(patient):

    data = pd.read_parquet('data/respiration/{patient}_all_respiration_data.parquet'.format(patient=patient))

    patient_info = Patient.patient_class(patient)
    patient_info.get_seizure_annotations()
    patient_info.seizure_table

    # data['datetime'] += pd.Timedelta(patient_info.patient_dict['temporal_shift'])
    return data


def get_training_segments(data, slide, window):
    """
    Get training segments from the data
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data with the respiratory signal
    slide : int
        Slide of the segments
    window : int
        Window of the segments
    
    Returns
    -------
    train_segments : pandas.DataFrame
        Training segments"""

    timestamps_segments = pd.date_range(start=data['datetime'].iloc[0], end=data['datetime'].iloc[-1]-pd.Timedelta(seconds=60), freq=f'{slide}s')
    # only use 20% of the data for training
    timestamps_segments_train_limit = timestamps_segments[0] + (timestamps_segments[-1] - timestamps_segments[0]) * 0.2
    timestamps_segments_train = timestamps_segments[timestamps_segments<timestamps_segments_train_limit]
    train_data = data.loc[data['datetime'].between(timestamps_segments_train[0], timestamps_segments_train[-1])].copy()

    train_path = f'preepiseizures{os.sep}sudep_analysis{os.sep}{patient}_data_segments_train_20p_{slide}s.parquet'
    train_segments = Respiration_2023.epilepsy_dataset(train_data, timestamps_segments_train, train_path, slide=slide, window=window)
    if train_segments.empty:
        return pd.DataFrame(), pd.Timestamp('NaT')
    return train_segments, timestamps_segments_train_limit


def get_validation_segments(data, timestamps_segments_train_limit, window):
    """
    Get validation segments from the data
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data with the respiratory signal
    
    Returns
    -------
    validation_segments : pandas.DataFrame
        Validation segments"""

    timestamps_segments_validation = pd.date_range(start=timestamps_segments_train_limit, end=data['datetime'].iloc[-1]-pd.Timedelta(seconds=60), freq='60s')
    validation_data = data.loc[data['datetime'].between(timestamps_segments_validation[0], timestamps_segments_validation[-1])].copy()
    validation_segments = Respiration_2023.epilepsy_dataset(validation_data, timestamps_segments_validation, 
                                                            train_path=f'preepiseizures{os.sep}sudep_analysis{os.sep}{patient}_validation_data_60s_timecorrected.parquet', 
                                                            slide=60, window=window)
    return validation_segments


def correlation_points(output_val, validation_segments):
    """
    Get correlation points from the output of the autoencoder and the validation segments
    
    Parameters
    ----------
    output_val : numpy.array
        Output of the autoencoder
    validation_segments : pandas.DataFrame
        Validation segments
    """
    
    corr_ = np.array(list(map(lambda i: Respiration_2023.correlation(output_val[i], validation_segments.iloc[i], points=1), range(len(output_val)))))
    corr_df = pd.DataFrame(corr_, index= validation_segments.index, columns=[str(i) for i in range(len(corr_[0]))])
    time_corr_points = [pd.date_range(corr_df.index[i], corr_df.index[i]+pd.Timedelta(seconds=60), periods=1) for i in range(len(corr_df.index))]
    corr_points = pd.DataFrame(np.hstack(corr_df.values), index=np.hstack(time_corr_points))
    return corr_points


if __name__ == '__main__':


    patient_list = [file.split('_')[0] for file in os.listdir('data/respiration') if file.endswith('all_respiration_data.parquet')]

    slide, window = 1, 60
    for patient in patient_list:
        if patient in ['YWJN', 'BLIW']:
            continue
        print('Processing patient: ', patient)
        corr_points_file = f'data{os.sep}autoencoders_epilepsy{os.sep}{patient}_corr_points_{slide}s.parquet'
        #if os.path.exists(corr_points_file):
        #    print('Correlation points already calculated for patient: ', patient)
        #    continue
        # GET PATIENT DATA -----
        data = get_patient_data(patient)
        data.sort_values(by='datetime', inplace=True)
        data.dropna(inplace=True)
        if data.empty:
            continue

        # TRAINING SEGMENTS ----
        train_segments_file = f'data/autoencoders_epilepsy{os.sep}{patient}_data_segments_train_20p_{slide}s.parquet'
        if os.path.exists(train_segments_file):
            train_segments = pd.read_parquet(train_segments_file)
            timestamps_segments_train_limit = train_segments.index[-1]
        else:
            train_segments, timestamps_segments_train_limit = get_training_segments(data, slide, window)
        if train_segments.empty:
            print('No training segments for patient: ', patient)
            continue
        # TRAINING AUTOENCODER -
        label = f'data{os.sep}autoencoders_epilepsy{os.sep}{patient}_{slide}s'
        modelAE, encAE, decAE = Respiration_2023.respiration_training(train_segments, again=False, label=label)

        # VALIDATION DATA ------
        validation_segments_file = f'data/autoencoders_epilepsy{os.sep}{patient}_validation_data_60s_timecorrected.parquet'
        if os.path.exists(validation_segments_file):
            validation_segments = pd.read_parquet(validation_segments_file)
        else:
            validation_segments = get_validation_segments(data, timestamps_segments_train_limit, window)

        output_val = modelAE.predict(validation_segments)

        # IO Correlation --------
        corr_points = correlation_points(output_val, validation_segments)
        corr_points.rename(columns={0:'corr'}, inplace=True)
        corr_points.to_parquet(corr_points_file)
        
        

