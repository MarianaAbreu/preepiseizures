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
    sensor = 'ecg'
    source = 'hospital'
    device = 'chestbit'
    patient_json = json.load(open(os.path.join('patient_info.json'), 'r'))

    for patient_id in patient_json.keys():

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
        check = biosignal.get_ecg_hospital_data(patient_class=patient)
        if check == 1:
            print('ECG data loaded.')
        else:
            print('ECG data not loaded.')
            continue
        print('Calculating ECG segments...\n')
        biosignal.calc_ecg_hospital_segments(patient_class=patient)
        pass
        

