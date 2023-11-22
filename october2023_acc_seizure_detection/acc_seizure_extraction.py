# %% [markdown]
# # 1. Analysis of Seizures with Motion using Chest and Wrist ACC
## Created by: Mariana Abreu
# Last modified: 20/09/2023
# %%
#built-in
import datetime
import json
import os

#external
import numpy as np
import pandas as pd

#local
from preepiseizures.src import OpenFileProcessor, Patient


def find_acc_patients():
    """
    Find patients with seizures with motor behaviour
    
    Parameters:
        None
    Returns:
        list of patients with annotated seizures with motor behaviour
    """
    patient_dict = json.load(open('patient_info.json'))
    patient_acc_list = []

    for patient_one in patient_dict.keys():

        patient = Patient.patient_class(patient_one)
        if not patient:
            continue
        patient.get_seizure_annotations()
        if not patient.seizure_table.empty:
            if ('motor' in str(patient.seizure_table.values) or 'FBTC' in str(patient.seizure_table.values) or 'automatismos' in str(patient.seizure_table.values).lower()):
                patient_acc_list.append(patient_one)
                # print(f"{patient_one} -- added to the list \n {patient.seizure_table} ------------------")
            # else:
                # print(f"{patient_one} -- excluded for lack of motor behaviour \n ------------------")
    return patient_acc_list


if __name__ == "__main__":
    
    # returns patients with seizures with motor behaviour and wearable data
    
    patient_list = ['AGGA', 'BLIW', 'VNVW', 'WOSQ','YIVL','YWJN']
    sensor = 'acc'
    type_ = 'all'

    for source in ['wearable']:

        for patient in patient_list:

            if os.path.isfile(f'./data/{patient}_{sensor}_{type_}_{source}.parquet'):
                continue
            else:
                print('Processing patient: ', patient)
            
            patient_ = Patient.patient_class(patient)
            patient_.get_seizure_annotations()
            if type_ == 'seizures':
                if source == 'wearable':
                    data = patient_.get_wearable_data_around_seizures(sensor)
                elif source == 'hospital':
                    data = patient_.get_hospital_data_around_seizures(sensor)
            if type_ == 'baseline':
                if source == 'wearable':
                    data = patient_.get_wearable_data_baseline(sensor)
                elif source == 'hospital':
                    # TODO: implement get_hospital_data_baseline
                    pass

            if data.empty:
                continue
            # data.to_parquet(f'./data/{patient}_{sensor}_seizures.parquet')
            data.to_parquet(f'./data/{patient}_{sensor}_{type_}_{source}.parquet')
            