# Seizure Prediction Framework for ECG Signals
#
# Author:  Mariana Abreu
# Date:    11/01/2024


import os

import numpy as np
import pandas as pd




class DatasetPred:

    """
    Class to load the dataset for prediction
    Receives data as parameter which should have features, a temporal instant associated to each sample/row, and the seizure onsets (in a column named "onset")
    """

    def __init__(self, data) -> None:

        self.data = data
        self.features = None

    def load_data(self, patient_info, data):
        """
        If the data is not in the desired format use this function to load or to check
        """

        # add datetime column as index
        if 'datetime' in data.columns:
            data = data.set_index('datetime')
        # add onset column if necessary
        if 'onset' in data.columns:
            pass
        else:
            data['onset'] = 0
            previous_onset = pd.Timestamp('1900-01-01 00:00:00')
            # run through the seizures
            for o in range(len(patient_info.seizure_table)):
                # get the seizure onset
                onset = patient_info.seizure_table.iloc[o]['Timestamp']
                # get the closest timestamp in the data
                onset_dateindex = data.index[np.argmin(abs(data.index - onset))]
                # check if the timestamp is in the seizure vicinity
                if (onset_dateindex - onset) < pd.Timedelta('30s'):
                    # check if seizure is near the last seizure
                    if (onset_dateindex - previous_onset) < pd.Timedelta(hours=2):
                        # if it is, mark it as the same seizure == seizure cluster
                        data.loc[onset_dateindex, 'onset'] = data.loc[previous_onset, 'onset']
                        print('Seizure cluster was at : {}'.format(onset_dateindex - previous_onset))
                    else:
                        # if it is not, mark it as a new seizure
                        data.loc[onset_dateindex, 'onset'] = o + 1
                    # update the previous onset to compare with the next seizure
                    previous_onset = onset_dateindex
                else:
                    print('Seizure {} not found in data'.format(o + 1))
                    print('Closest timestamp was at : {}'.format(onset_dateindex - onset))
                    continue
        self.data = data
        print('Data loaded successfully')
        assert type(self.data.index) == pd.core.indexes.datetimes.DatetimeIndex, "The index of the data should be a datetimeindex"
        assert self.data.index.name == 'datetime', "The index of the data should be named datetime"
        return data
    
    def remove_periods_data(self, data, preictalmax=120, posictalmax=7200):
        """
        Remove the data from the periods ictal and postictal 
        :param data: dataframe with the data
        :param preictalmax: maximum time in seconds before the seizure onset
        :param posictalmax: maximum time in seconds after the seizure onset
        """
        # get seizure onsets
        onsets = data['onset'].unique()
        # create a copy of the data
        new_data = data.copy()
        # place the time as a column
        new_data.reset_index(inplace=True)
        # create a list to store the indexes to be removed
        remove_indexes = []
        for o in range(1, len(onsets)):
            # get the seizure onset
            onset = new_data.loc[new_data['onset'] == o]['datetime'].iloc[0]
            # get the indexes of the ictal and postictal periods (preictalmax seconds before the seizure will also be removed)
            remove_indexes.append(new_data.loc[new_data['datetime'].between(onset - pd.Timedelta(seconds=preictalmax), onset + pd.Timedelta(seconds=posictalmax))].index)
            new_data.loc[remove_indexes[-1][0]-1, 'onset'] = onsets[o]
        # flatten the list
        remove_indexes = np.hstack(remove_indexes)
        # remove the indexes from the data
        new_data.drop(remove_indexes, inplace=True)
        new_data.set_index('datetime', inplace=True)
        return new_data

    def train_test_split(self, data, train_sz):
        """
        Split the dataset into train and test putting 3 seizures in the training
        All data before the third seizure will also be moved to the test set
        :param data: dataframe with the data with datetimeindex
        :param train_sz: number of seizures to be used in the training
        """

        # The onset column will have zeros for all samples except the existing seizures which will be marked as 1, 2, 3, ...
        # The number of each onset will respect the number of the seizure.
        # If one seizure is too close to another, it will be marked as the same number
        # If one seizure does not exist, the number will be jumped

        assert type(data.index) == pd.core.indexes.datetimes.DatetimeIndex, "The index of the data should be a datetimeindex"

        # print("The whole dataset contains {} seizures".format(data["onset"].max()))
        new_data = data.reset_index().copy()
        # The train set will have the first 3 seizures (which can be 1,2,3 or 1,2,5 as example)
        # Since 0 will also ocupy its space, the train set will end in the position 3
        assert len(new_data["onset"].unique()) > train_sz, "The number of seizures in the dataset is smaller than the number of seizures to be used in the training"
        third_seizure_onset = new_data.loc[new_data['onset'] == new_data["onset"].unique()[train_sz]].index[0]
        # train e test datasets
        x_train = new_data.loc[:third_seizure_onset]
        x_test = new_data.loc[third_seizure_onset:]
        if 'datetime' in x_train.columns:
            x_train.set_index('datetime', inplace=True)
        if 'datetime' in x_test.columns:
            x_test.set_index('datetime', inplace=True)

        return x_train, x_test
        
    def annotate_data(self, data, preictal0 = 45, preictal1 = 15):
        """
        Annotate the data into interictal and preictal
        :param data: dataframe with the data with datetimeindex called "datetime"
        :param patient_info: patient information
        :param preictal0: start of the preictal period in minutes
        :param preictal1: end of the preictal period in minutes
        """
        assert data.index.name == 'datetime', "The index of the data should be named datetime"

        data['Y'] = 'interictal'
        data.reset_index(inplace=True)
        # get seizure onsets
        onsets = data['onset'].drop_duplicates()
        for onset in onsets.iloc[1:].index:
            data.loc[data['datetime'].between(data['datetime'][onset] - pd.Timedelta(minutes=preictal0), 
                                              data['datetime'][onset] - pd.Timedelta(minutes=preictal1)), 'Y'] = 'preictal'
            data.loc[data['datetime'].between(data['datetime'][onset] - pd.Timedelta(minutes=preictal1), 
                                              data['datetime'][onset] + pd.Timedelta(minutes=120)), 'Y'] = 'removefromtrain'
            
        data.set_index('datetime', inplace=True)
        return data

class PredictionEval():

    """
    Class to evaluate the prediction results
    
    """

    def __init__(self) -> None:

        pass

    def predict_method(self, model, test_data):
        """
        Predict the labels of the data
        :param model: model to be used
        :param test_data: data to be predicted
        """
        # can be more complicated than calling predict, in the case where consecutive labels are evaluated together
        y_pred = model.predict(test_data)
        assert type(y_pred.index) == pd.core.indexes.datetimes.DatetimeIndex, "The index of the data should be a datetimeindex"
        # y_pred should have one column with 0 and 1s where 1s should be the alarms raised
        return y_pred

    def evaluate(self, seizure_onsets, y_pred, sop=15, sph=30):
        """
        Evaluate the model using the test data
        :param model: model to be evaluated
        :param test_data: test data should have index as datetime
        :param sop: seizure occurrence period in minutes
        :param sph: seizure prediction horizon in minutes
        """
        # test data has column "onset" where the seizure onsets are marked by order of occurrence


        # x_test = test_data.drop(columns=['Y'])
        
        # y_pred = self.predict_method(model, x_test)
        alarms = y_pred.loc[y_pred['Y'] == 1].copy().index

        true_positives = 0
        false_positives = 0

        seizures_detected = []
        seizures_missed = []

        for alarm in alarms:
            # check if any seizure onset is in the seizure occurrence period
            alarm_sop_start = alarm + pd.Timedelta(minutes=sph)
            alarm_sop_end = alarm + pd.Timedelta(minutes=sph) + pd.Timedelta(minutes=sop)
            # check if any onset is in the sop period
            if any((seizure_onsets >= alarm_sop_start) & (seizure_onsets <= alarm_sop_end)):
                true_positives += 1
                seizures_detected.append(seizure_onsets.between(alarm_sop_start, alarm_sop_end).index)
                
            else:
                false_positives += 1
            
        seizures_missed = seizure_onsets.drop(seizures_detected)
        false_negatives = len(seizures_missed)

        return true_positives, false_positives, false_negatives, seizures_detected, seizures_missed

    def evaluate_metrics(self, TP=0, FP=0, FN=0, sph=30, seizures_detected=[], seizures_missed=[], total_duration=None):
        """
        Metrics explained in: 
        - "Performance metrics for online seizure prediction", Hsiang-Han Chen, Vladimir Cherkassky, https://doi.org/10.1016/j.neunet.2020.04.022
        Evaluate the model using the test data
        :param TP: true positives
        :param FP: false positives
        :param FN: false negatives
        :param seizures_detected: list of seizures detected
        :param seizures_missed: list of seizures missed
        :param total_duration: total duration in hours of the interictal test data
        :return: sensitivity, far, specificity
        """
        assert total_duration is not None, "The total duration of the data must be given"
        # sensitivity is the proportion of seizures detected from the total number of seizures
        sensitivity = TP / (TP + FN)
        # specificity is the proportion of non-seizures detected from the total number of non-seizures
        # There are two performance indices related to specificity: 
        # time in warning (TIW) and false positive error rate (FPR) 
        # get far ou fpr for 24 hours
        far = (FP / total_duration) 
        # TIW is defined as the fraction of time the system makes positive (preictal) predictions,
        # or the system is in warning state
        tiw = (TP + FP) * sph / total_duration


        return sensitivity, far, tiw



