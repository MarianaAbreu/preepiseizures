#
# Get physiological information around seizures:
#     1) get seizure annotations 
#     2) get data around seizures
#     3) get ecg and resp analysis
#     4) save everything in a csv file per patient
# Created: 15.07.2023
# By: Mariana Abreu
#
#built-in
import os
from datetime import datetime

#external
import biosppy as bp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#local
from src import Patient

if __name__ == "__main__":

    # main_dir = '/Volumes/My Passport/Patients_HEM/Retrospective'
    main_dir = '/Volumes/T7 Touch/PreEpiSeizures/Patients_HEM'

    for patient in os.listdir(main_dir):
        if os.path.isfile('seiz_info_' + patient + '.csv'):
            continue
        if patient.startswith('.'):
            continue
        if not os.path.isdir(os.path.join(main_dir, patient)):
            continue
        print('Extracting seizure info for patient: ', patient)
        patient_data = Patient.Patient(patient, main_dir)
        try:
            patient_data.get_seizure_annotations()
        except:
            print('No seizure annotations found for patient: ', patient)
            continue
        filesdir = [dir for dir in os.listdir(patient_data.dir) if os.path.isdir(os.path.join(patient_data.dir, dir))]
        filesdir = [dir for dir in filesdir if '.trc' in str(os.listdir(os.path.join(patient_data.dir, dir))).lower()]
        if len(filesdir) >= 1:
            filesdir = os.path.join(patient_data.dir, filesdir[0])
        #elif len(filesdir) > 1:
            #   for filedir in filesdir:
            #    filedir = os.path.join(patient_data.dir, filedir)
            #    list_dates[filedir] = patient_data.get_file_start_date('HEM', filedir)
        list_dates = patient_data.get_file_start_date('HEM', filesdir)
        all_df = patient_data.get_all_seizures_df(list_dates)

        seiz_info = pd.DataFrame()
        
        for sidx in range(len(patient_data.seizure_table)):
            # get preictal info
            seizure_date = datetime.strptime(patient_data.seizure_table.iloc[sidx]['Date'], '%d-%m-%Y\n%H:%M:%S')
            preictal_info = patient_data.get_seizure_info(all_df, seizure_date, sidx, 'preictal')
            preictal_info.update(patient_data.seizure_table.iloc[sidx].to_dict())
            seiz_info = pd.concat([seiz_info, pd.DataFrame(preictal_info, index=[0])], ignore_index=True)
            ictal_info = patient_data.get_seizure_info(all_df, seizure_date, sidx, 'ictal')
            ictal_info.update(patient_data.seizure_table.iloc[sidx].to_dict())
            posictal_info = patient_data.get_seizure_info(all_df, seizure_date, sidx, 'posictal')
            posictal_info.update(patient_data.seizure_table.iloc[sidx].to_dict())
            seiz_info = pd.concat([seiz_info, pd.DataFrame(ictal_info, index=[0])], ignore_index=True)
            seiz_info = pd.concat([seiz_info, pd.DataFrame(posictal_info, index=[0])], ignore_index=True)
        
        seiz_info.to_csv('seiz_info_' + patient + '.csv')


