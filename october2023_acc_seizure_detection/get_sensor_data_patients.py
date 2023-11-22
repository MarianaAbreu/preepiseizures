# Get data from one sensor for the desired patients
#
# Created by: Mariana Abreu
# Created on: 18/10/2023
# Last modified: 18/10/2023

# built-in modules
import os
import json

# external modules
import numpy as np
import pandas as pd

# local modules
from preepiseizures.src import Patient
from preepiseizures.src import Biosignal


if __name__ == '__main__':

    patient_list = ['YIVL']
    seizures_dict = {'AGGA': [0, 1, 2, 3], 'BLIW': [1, 2, 3], 'VNVW': [1], 'WOSQ': [2], 'YIVL': [0, 1, 2], 'YWJN': [4, 5, 6, 7]}
    sensor = 'acc'
    source = 'wearable'
    device = 'chestbit'

    for patient_id in patient_list:

        print(f'\nProcessing patient {patient_id}...')
        try:
            patient = Patient.patient_class(patient_id)
        except:
            print(f'Patient {patient_id} not found.')
            continue
        biosignal = Biosignal.Biosignal(sensor, device, source)
        # get only seizures FBTC
        seizures_idx = seizures_dict[patient_id]
        patient.get_seizure_annotations()
        seizures_select = patient.seizure_table.iloc[seizures_idx]
        # find files that contain these seizures using quality table
        quality = patient.get_quality_data()
        quality_ = Patient.correct_patient(quality, patient_id)
        
        if patient in ['BLIW', 'YWJN', 'YIVL']:
            temporal_shift = patient.patient_dict['temporal_shift']
        
        # get all files
        #files_seizures = np.hstack(list(map(lambda x: quality_.loc[quality_['datetime'].between(x-pd.Timedelta(hours=3), 
        #                                                                              x+pd.Timedelta(hours=2))]['filename'].unique(), 
        #                                                                              seizures_select['Timestamp'])))
        # find the respective filename in the wearable directory that is in files_seizures
        # files_seizures_txt = []
        # filepath = os.path.join(patient.dir, patient.patient_dict['wearable_dir'])
        # all_files = os.listdir(filepath)
        # for filename in files_seizures:
        #     if os.path.isfile(os.path.join(filepath, filename + '.txt')):
        #         files_seizures_txt.append(filename + '.txt')
        #     else:
        #         print(f'File {filename} not found.')
        #         pass
        files_seizures_txt = [file for file in os.listdir(os.path.join(patient.dir, patient.patient_dict['wearable_dir'])) if (file.endswith('.txt') and file.startswith('A2'))]
        if sensor == 'acc':
            biosignal.calc_acc_wearable_features(patient_class=patient, files_list=files_seizures_txt)
        pass
        

