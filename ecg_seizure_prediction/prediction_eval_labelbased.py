# Script to run label-based evaluation of predictions
# Input data is the HRV features with hour and time of day columns
# Output is a csv file with the evaluation metrics for each prediction

# Import libraries
# built-in
import json
import os

# third-party
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM

# local
from preepiseizures.src import Patient, Prediction, biosignal_processing
import prediction_run_features as prf


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Generate some example data (replace this with your actual data)
# X should be your feature matrix, y should be your labels (1 for normal, -1 for anomaly)
TRAIN_SZ = 3

columns_to_drop = ['Y', 'daypart', 'onset']


def dataset_preparation(dataset, annotated_dataset, train_sz, columns_to_drop, scaler, remove_from_train=True):
    """
    This function prepares the dataset for training and testing.
    
    Parameters
    ----------
    annotated_dataset : pd.DataFrame
        The annotated dataset.
    train_sz : int
        The number of seizures to use for training.
    columns_to_drop : list
        The columns to drop from the dataset.
    scaler : sklearn.preprocessing.MinMaxScaler
        The scaler to use for the data.
    
    Returns
    -------
    x_train_final : pd.DataFrame
        The training dataset.
    x_test_final : pd.DataFrame
        The testing dataset.
    y_train : pd.Series
        The training labels.
    y_test : pd.Series
        The testing labels.
    """
    annotated_dataset['hourpart'] = annotated_dataset.index.hour
    # drop inf values
    if 'fbands' in annotated_dataset.columns:
        annotated_dataset.drop(columns=['fbands'], inplace=True)
    annotated_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    annotated_dataset.dropna(inplace=True)

    x_train, x_test = dataset.train_test_split(annotated_dataset, train_sz = train_sz)
    if remove_from_train:
        x_train_usable = x_train.loc[x_train['Y'] != 'removefromtrain'].copy()
        x_test_usable = x_test.loc[x_test['Y'] != 'removefromtrain'].copy()
    else:
        x_train_usable = x_train.copy()
        x_test_usable = x_test.copy()

    scaler.fit(x_train_usable.drop(columns=columns_to_drop))
    x_train_scaled = scaler.transform(x_train_usable.drop(columns=columns_to_drop))
    
    x_test_scaled = scaler.transform(x_test_usable.drop(columns=columns_to_drop))
    x_train_final = pd.DataFrame(x_train_scaled, index=x_train_usable.index, 
                                columns=x_train_usable.drop(columns=columns_to_drop).columns)
    x_train_final['daypart'] = x_train_usable['daypart'].copy()

    x_test_final = pd.DataFrame(x_test_scaled, index=x_test_usable.index, 
                                columns=x_test_usable.drop(columns=columns_to_drop).columns)

    x_test_final['daypart'] = x_test_usable['daypart'].copy()

    y_train = x_train_usable['Y'].copy()
    y_test = x_test_usable['Y'].copy()

    return x_train_final, x_test_final, y_train, y_test


def main(dataset, annotated_dataset, scaler, Nfeat, model_name, feature_selection_bool=True, show_plot=False):
    # Split the data into training and testing sets

    X_train, X_test, y_train, y_test = dataset_preparation(dataset, annotated_dataset, train_sz=TRAIN_SZ, 
                                                           columns_to_drop=columns_to_drop, scaler=scaler, remove_from_train=True)
    y_train = (y_train == 'preictal').astype(int)
    y_test = (y_test == 'preictal').astype(int)

    if feature_selection_bool:
       # Feature selection using SelectKBest with ANOVA F-statistic
        k_best = SelectKBest(f_classif, k=Nfeat)  # Adjust the number of features based on your needs
        X_train_kbest = k_best.fit_transform(X_train, y_train)
        X_test_kbest = k_best.transform(X_test)
    else:
        X_train_kbest = X_train
        X_test_kbest = X_test


    if model_name == 'IF':
        # Train the model on the selected features
        model = IsolationForest(contamination=0.1, random_state=42)
    elif model_name == 'SVM':
        model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')  # Adjust parameters as needed

    model.fit(X_train_kbest)

    # Predict the anomaly scores on the test set
    y_scores = model.decision_function(X_test_kbest)

    # Compute ROC curve and ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    if show_plot:
        # Plot the ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve with Feature Selection')
        plt.legend(loc='lower right')
        plt.show()

    dict_eval = {}
    dict_eval['AUC'] = np.round(roc_auc, 3)
    dict_eval['Nfeat'] = Nfeat
    dict_eval['model'] = model_name
    dict_eval['model_params'] = str(model.get_params())
    if feature_selection_bool:
        dict_eval['feature_selection'] = k_best.get_feature_names_out()
    else:
        dict_eval['feature_selection'] = 'All features'

    dict_eval['train_sz'] = TRAIN_SZ

    return dict_eval


def dict_eval_join(dataset, annotated_dataset, scaler, Nfeat, model_name, feature_selection_bool=True, show_plot=False, patient_info=None):
    """
    This function outputs a dataframe with the evaluation metrics for each prediction.
    """
    dict_eval = main(dataset, annotated_dataset, scaler, Nfeat, model_name, feature_selection_bool=feature_selection_bool, show_plot=show_plot)
    dict_eval['patient'] = patient
    dict_eval['seizures'] = len(patient_info.seizure_table)
    # make it a string to avoid errors when saving to a dataframe
    dict_eval['seizure_types'] = ' '.join(patient_info.seizure_table['Focal / Generalisada'].astype(str).values)
    dict_eval['feature_selection'] = ' '.join(dict_eval['feature_selection'])
    table = pd.DataFrame(dict_eval, index=[0])
    return table


if __name__ == '__main__':

    # SELECT PATIENT ----------
    patient_json = json.load(open('patient_info.json'))
    
    for patient in patient_json.keys():
        final_results_file = os.path.join('data', 'prediction', patient + '_prediction_eval_labelbased.csv')

        if os.path.exists(final_results_file):
            print('Patient already processed: ', patient)
            continue
        
        failed_file = os.path.join('data', 'prediction', patient + '_prediction_eval_labelbased_failed.txt')
        if os.path.exists(failed_file):
            print('Patient already failed: ', patient)
            continue

        print('Processing patient: ', patient)
        try:
            patient_info = Patient.patient_class(patient)
        except:
            patient_info = None
        if not patient_info:
            message = f'Patient {patient} info not found'
            with open(failed_file, 'w') as f:
                f.write(message + '\n')
            continue
        if patient_info.patient_dict['hospital_dir'] == '':
            message = f'Patient {patient} hospital files not found'
            with open(failed_file, 'w') as f:
                f.write(message + '\n')
            continue
        check_sz = patient_info.get_seizure_annotations()
        if check_sz == -1:
            message = f'Patient {patient} seizure annotations not found'
            with open(failed_file, 'w') as f:
                f.write(message + '\n')
            continue

        # GET HRV DATA ------------
        patient_hrv_name = os.path.join('data', 'prediction', patient + '_hrv_data.pkl')
        if not os.path.exists(patient_hrv_name):
            patient_data = os.path.join('data', 'prediction', patient + '_data_all.pkl')

            data = prf.fetch_data_prepared(patient_data, patient_info, quality=True, rpeaks=True, save=True)
            if data.empty:
                message = f'Patient {patient} original data not loaded'
                with open(failed_file, 'w') as f:
                    f.write(message + '\n')
                continue
            try:
                hrv_data = biosignal_processing.get_hrv_train(data)
                hrv_data.to_pickle(patient_hrv_name)
            except Exception as e:
                print(e)
                message = f'Patient {patient} HRV data not loaded with error {e}'
                with open(failed_file, 'w') as f:
                    f.write(message + '\n')
                continue

        # LOAD DATASET ------------
        data_final = pd.read_pickle(patient_hrv_name)
        data_final.index.name = 'datetime'
        dataset = Prediction.DatasetPred(data=data_final)
        predictor = Prediction.PredictionEval()
        dataset_data = dataset.load_data(patient_info=patient_info, data=data_final)
        annotated_dataset = dataset.annotate_data(data= dataset_data)

        # if feat is nan, remove from annotated dataset
        annotated_dataset = annotated_dataset.dropna(axis=1)
        if annotated_dataset.empty:
            message = f'Patient {patient} annotated dataset not loaded'
            with open(failed_file, 'w') as f:
                f.write(message + '\n')
            continue
        try:
            x_train, x_test = dataset.train_test_split(annotated_dataset, train_sz = 3)
        except Exception as e:
            print(e)
            message = f'Patient {patient} train test split failed with error {e}'
            with open(failed_file, 'w') as f:
                f.write(message + '\n')
            continue
        # EVAL MODELS -------------    
        table_all = pd.DataFrame()

        for model_name in ['IF', 'SVM']:
            for Nfeat in range(2, len(annotated_dataset.columns)-2):
                for scaler in [MinMaxScaler(), StandardScaler()]:
                    table = dict_eval_join(dataset, annotated_dataset, scaler, Nfeat, 
                                        model_name, feature_selection_bool=True, 
                                        show_plot=False, patient_info=patient_info)
                    table_all = pd.concat((table_all, table))

        # SAVE TABLE --------------
        table_all.sort_values(by='AUC', ascending=False, inplace=True)
        table_all.to_csv(final_results_file, index=False)

