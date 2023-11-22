# ACC Seizure Detection
# 1) Separate into training and testing
# 2) Extract features and normalization
# This notebook gets features from 3s intervals and then gets the median value for 120 seconds
# 3) Train model
# 4) Test model
# Created by: Mariana Abreu
# Creation date: 4 September 2023
# Last update: 07/09/2023
# built-in
import datetime
import json
import os
from random import randrange
import random

# external
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# local
from preepiseizures.src import Patient


def train_test_separation():
    """
    Separate patients into training and testing
    """
    
    # TODO

    pass

def extract_features(sensors_data_labels, rule='3s', cols_excluded = ['datetime', 'filename']):
    """
    Extract features from training/testing data segment
    """
    # get segments with 3 seconds
    segments = sensors_data_labels.resample(rule=rule)
    # get features
    # mean, std, median, kurtosis, skewness, minmax
    cols = [col for col in sensors_data_labels.columns if col not in cols_excluded]
    df_features = pd.DataFrame()

    df_features['mean_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].mean()
    df_features['median_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].median()
    df_features['std_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].std()
    df_features['kurt_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].apply(pd.DataFrame.kurt)
    df_features['skew_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].skew()
    df_features['minmax_' + sensors_data_labels.columns.drop(cols_excluded)] = segments[cols].max() - segments[cols].min()
    if not df_features.empty:
        df_features['relative_time'] = df_features.index - sensors_data_labels.index[0]
        df_features['onset'] = sensors_data_labels.index[0]
    # join labels to features
    
    return df_features



def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + datetime.timedelta(seconds=random_second)



def extract_seizures_2(patient, sensors_data_labels, rule='3s', cols_excluded = ['datetime', 'filename'], ratio = True):
    """
    Extract features from training/testing data segment
    This function calculates features for X intervals after seizure onset where X=120 seconds/rule
    This intervals will have the size of rule
    """
    # get features

    if patient.seizure_table.empty:
        patient.get_seizure_annotations()
    # join labels to features
    # df_features['datetime'] = df_features.index
    all_ratio = pd.DataFrame()
    for sidx in patient.seizure_table.index:
        seiz_time = patient.seizure_table.loc[sidx, 'Timestamp']
        seizure_data = sensors_data_labels.loc[sensors_data_labels['datetime_chestbit'].between(seiz_time, seiz_time + pd.Timedelta(seconds=120))].copy()
        df_features = extract_features(seizure_data, rule=rule, cols_excluded=cols_excluded)
        
        df_features['seizure'] = sidx
        df_features['datetime'] = df_features.index
        df_features['onset'] = seiz_time
        df_features['seizure_type'] = patient.seizure_table.loc[sidx, 'Focal / Generalisada']
        df_features['seizure_subtype'] = patient.seizure_table.loc[sidx, 'Tipo']
        all_ratio = pd.concat([all_ratio, df_features])
        
    all_ratio = all_ratio.dropna(subset = [col for col in all_ratio.columns if col not in ['seizure', 'datetime', 'seizure_type', 'seizure_subtype']])
    all_ratio.index = all_ratio['datetime']
    return all_ratio


def extract_baseline_2(sensors_data_labels, n_seizures = 0, rule='3s', cols_excluded = ['datetime', 'filename'], ratio = True):
    """
    Extract features from training/testing data segment
    This function calculates features for X intervals after seizure onset where X=120 seconds/rule
    This intervals will have the size of rule
    """
    # get features

    # join labels to features
    # df_features['datetime'] = df_features.index
    # all_ratio = pd.DataFrame()
    # for sidx in patient.seizure_table.index:
    #     seiz_time = patient.seizure_table.loc[sidx, 'Timestamp']
    #     seizure_data = sensors_data_labels.loc[sensors_data_labels['datetime_chestbit'].between(seiz_time, seiz_time + pd.Timedelta(seconds=120))].copy()
    #     df_features = extract_features(seizure_data, rule=rule, cols_excluded=cols_excluded)
        
    #     df_features['seizure'] = sidx
    #     df_features['datetime'] = df_features.index
    #     df_features['onset'] = seiz_time
    #     df_features['seizure_type'] = patient.seizure_table.loc[sidx, 'Focal / Generalisada']
    #     df_features['seizure_subtype'] = patient.seizure_table.loc[sidx, 'Tipo']
    #     all_ratio = pd.concat([all_ratio, df_features])
        
    # all_ratio = all_ratio.dropna(subset = [col for col in all_ratio.columns if col not in ['seizure', 'datetime', 'seizure_type', 'seizure_subtype']])
    # all_ratio.index = all_ratio['datetime']
    # df_features = extract_features(sensors_data_labels, rule=rule, cols_excluded=cols_excluded)

    # # join labels to features
    # df_features['datetime'] = df_features.index
    # all_ratio = pd.DataFrame()
    # run through the df_features as many times as there are seizures

    baseline_segments = sensors_data_labels.resample(rule='2T')
    df_features = baseline_segments.apply(extract_features, rule=rule)

    if len(df_features) > n_seizures:
        
        idxs = random.choices(range(len(df_features['onset'].unique())), k=n_seizures)
        idxs_segments = df_features['onset'].unique()[idxs]
        all_ratio = df_features.loc[df_features['onset'].isin(idxs_segments)].copy()
    else:
        all_ratio = df_features.copy()
    all_ratio['seizure'] = 0
    all_ratio['seizure_type'] = ''
    all_ratio['seizure_subtype'] = ''
 
    return all_ratio

def extract_ratio_seizures(patient, sensors_data_labels, rule='3s', cols_excluded = ['datetime', 'filename'], ratio = True):
    """
    Extract features from training/testing data segment
    """
    # get features
    df_features = extract_features(sensors_data_labels, rule=rule, cols_excluded=cols_excluded)
    if patient.seizure_table.empty:
        patient.get_seizure_annotations()
    # join labels to features
    df_features['datetime'] = df_features.index
    all_ratio = pd.DataFrame()
    for sidx in patient.seizure_table.index:
        seiz_time = patient.seizure_table.loc[sidx, 'Timestamp']

        df_seizure = df_features.loc[df_features['datetime'].between(seiz_time, seiz_time + pd.Timedelta(seconds=120))].copy()
        if df_seizure.empty:
            continue
        else:
            df_seizure.drop(columns=['datetime'], inplace=True)
            if ratio:
                df_before_seizure = df_features.loc[df_features['datetime'].between(seiz_time - pd.Timedelta(hours=1), seiz_time - pd.Timedelta(minutes=20))].dropna()
                df_before_seizure.drop(columns=['datetime'], inplace=True)
                median_ratio = df_seizure.median() / df_before_seizure.median()

                if not median_ratio.loc[median_ratio==np.inf].empty:
                    zeros_vals = median_ratio.loc[median_ratio==np.inf].index
                    median_ratio[zeros_vals] = df_seizure.median()[zeros_vals]
            else:
                median_ratio = df_seizure.median()
            median_ratio['seizure'] = sidx
            median_ratio['datetime'] = seiz_time
            median_ratio['seizure_type'] = patient.seizure_table.loc[sidx, 'Focal / Generalisada']
            median_ratio['seizure_subtype'] = patient.seizure_table.loc[sidx, 'Tipo']
            all_ratio = pd.concat([all_ratio, median_ratio], axis=1)
        
    all_ratio = all_ratio.T.dropna(subset = [col for col in all_ratio.T.columns if col not in ['seizure', 'datetime', 'seizure_type', 'seizure_subtype']])
    all_ratio.index = all_ratio['datetime']
    return all_ratio

def extract_ratio_baseline(sensors_data_labels, rule='3s', cols_excluded = ['datetime', 'filename'], ratio = True, n_seizures=0):
    """
    Extract features from training/testing data segment
    """
    # get features
    df_features = extract_features(sensors_data_labels, rule=rule, cols_excluded=cols_excluded)

    # join labels to features
    df_features['datetime'] = df_features.index
    all_ratio = pd.DataFrame()
    # run through the df_features as many times as there are seizures

    baseline_segments = df_features.resample(rule='2T')
    median_ratio = baseline_segments.median().dropna()

    if len(median_ratio) > n_seizures:
        idxs = random.choices(median_ratio.index, k=n_seizures)
        median_ratio = median_ratio.loc[idxs]

    median_ratio['seizure'] = 0
    median_ratio['seizure_type'] = ''
    median_ratio['seizure_subtype'] = ''
 
    return median_ratio


def get_patient_features(patient_name, rule):
    """
    Get patients features
    """

    patient = Patient.patient_class(patient_name)
    patient_acc_seizures = pd.read_parquet(f'./data/{patient_name}_acc_seizures.parquet')
    if patient_acc_seizures.empty:
        return pd.DataFrame()
    
    patient_acc_baseline = pd.read_parquet(f'./data/{patient_name}_acc_baseline.parquet')

    patient_acc_seizures.index = patient_acc_seizures['datetime_chestbit']
    cols_exclude = ['activity_index_chestbit', 'ACCX_wristbit', 'ACCY_wristbit', 'ACCZ_wristbit', 
                    'datetime_wristbit', 'activity_index_wristbit', 'acc_magnitude_wristbit', 
                    'datetime_chestbit', 'filename', 'patient', 'seizure']
    feats_seizure = extract_seizures_2(patient, patient_acc_seizures, cols_excluded=cols_exclude, ratio=False, rule=rule)
    col_dict = {col:col.split('_chestbit')[0] for col in feats_seizure}    
    n_seizures = len(feats_seizure['onset'].unique())
    feats_seizure.rename(columns=col_dict, inplace=True)
    patient_acc_baseline.index = patient_acc_baseline['datetime']
    cols_exclude = ['datetime', 'filename', 'activity_index']
    feats_baseline = extract_baseline_2(patient_acc_baseline, cols_excluded=cols_exclude, ratio=False, n_seizures=n_seizures, rule=rule)
    feats_seizure['Y'] = 1
    feats_baseline['Y'] = 0
    X = pd.concat([feats_seizure, feats_baseline], axis=0)
    return X


def get_data():
    """
    Get data from all patients
    """
    rule = '20S'

    #if os.path.exists(f'./data/X_acc_{rule}.parquet'):
    #    return pd.read_parquet(f'./data/X_acc_{rule}.parquet')
    
    patient_list = [pat.split('_')[0] for pat in os.listdir('./data/') if 'acc' in pat and 'seizures' in pat]
    all_X = pd.DataFrame()

    for patient in patient_list:
        print('GETTING PATIENT: ', patient)
        X = get_patient_features(patient, rule)
        if X.empty:
            continue
        X['patient'] = patient
        all_X = pd.concat([all_X, X], axis=0)

    all_X['datetime'] = all_X.index
    all_X.index = np.arange(len(all_X))
    all_X.drop(columns=['seizure'], inplace=True)
    all_X.to_parquet(f'./data/X_acc_{rule}.parquet')
    return all_X

if __name__ == '__main__':
    
    X = get_data()