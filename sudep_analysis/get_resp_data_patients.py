# Get data from one sensor for the desired patients
#
# Created by: Mariana Abreu
# Created on: 18/10/2023
# Last modified: 17/11/2023

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

    # patient_list = ['BLIW']
    # seizures_dict = {'AGGA': [0, 1, 2, 3], 'BLIW': [1, 2, 3], 'VNVW': [1], 'WOSQ': [2], 'YIVL': [0, 1, 2], 'YWJN': [4, 5, 6, 7]}
    sensor = 'pzt'
    source = 'wearable'
    device = 'chestbit'
    patient_json = json.load(open(os.path.join('patient_info.json'), 'r'))

    for patient_id in patient_json.keys(): #['BLIW', 'YWJN', 'VNVW', 'WOSQ', 'YIVL', 'AGGA']:
        if patient_id in ['QFRK', 'RGNI', 'GPPF']:
            continue
        print(f'\nProcessing patient {patient_id}...')
        try:
            patient = Patient.patient_class(patient_id)
        except Exception as e:
            print(e)
            continue
        if not patient:
            print(f'Patient {patient_id} not found.')
            continue
        biosignal = Biosignal.Biosignal(sensor, device, source)
        print('Calculating RESP segments...\n')
        biosignal.calc_resp_wearable_segments(patient_class=patient, filepath=f'data{os.sep}respiration{os.sep}{patient_id}_all_respiration_data.parquet')
        pass
        

