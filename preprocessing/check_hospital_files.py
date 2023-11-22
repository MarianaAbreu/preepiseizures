# Hospital files should be sequential between two numbers: e.g.: FA7775MK -> FA7775NR  
# Created by: Mariana Abreu
# Date: 26-07-2023
#
# built-in
import json
import os
import string

# external
import pandas as pd

# local
from src import Patient


def file_range(first_file, last_file):

    file_list = [first_file]
    alphabet = string.digits + string.ascii_uppercase 
    if first_file == '':
        return []
    i = alphabet.index(first_file[-1])
    j = alphabet.index(first_file[-2])
    while file_list[-1] != last_file:
        i += 1
        if file_list[-1][-1] != 'Z':
            new_file = file_list[-1][:-1] + alphabet[i]
        else:
            i = 0
            j += 1
            if file_list[-1][-2] != 'Z':
            # second row advances
                new_file = file_list[-1][:-2] + alphabet[j] + alphabet[i]
            else:
                j = 0
                if file_list[-1][-3] != 'Z':
                    new_file = file_list[-1][:-3] + alphabet[alphabet.index(file_list[-1][-3])+1] + alphabet[j] + alphabet[i]
                else:
                    new_file = file_list[-1][:-4] + alphabet[alphabet.index(file_list[-1][-4])+1] + alphabet[0] + alphabet[j] + alphabet[i]

        file_list.append(new_file)
    return file_list

def check_hospital_files(patient):

    hospital_dir = os.path.join(patient.dir, patient.patient_dict['hospital_dir'])
    formats = set([file.split('.')[-1] for file in os.listdir(hospital_dir)])
    format = [format for format in formats if format in ['TRC', 'edf', 'eeg']]
    # edf
    check_str = ''
    for form in sorted(format):
        
        if 'hospital_files' in patient.patient_dict.keys():
            first_file = patient.patient_dict['hospital_files'][0]
            last_file = patient.patient_dict['hospital_files'][1]
            file_list = file_range(first_file, last_file)
            current_list = [file.split('.')[0] for file in os.listdir(hospital_dir) if file.lower().endswith(form)]
            check_str += f'{len(current_list)}/{len(file_list)} {form} files '
        else:
            x = len([file for file in os.listdir(hospital_dir) if file.upper().endswith(format[0])])
            check_str = f'{x} {form} files'
    print(check_str)
    return check_str


if __name__ == '__main__':

    patient_dict = json.load(open('patient_info.json'))
    # select first patient as example
    df_check_everything = {}
    for patient_id in patient_dict.keys():
        print('Checking patient ', patient_id)
        df_check_everything[patient_id] = {}
        # sufix is HSM or HEM
        folder_dir = f"/Volumes/T7 Touch/PreEpiSeizures/Patients_{patient_dict[patient_id]['source']}"
        patient = Patient.Patient(patient_id, folder_dir)
        # --- check raw data files edf or eeg
        if patient.patient_dict['hospital_dir']:
            hospital_dir = os.path.join(patient.dir, patient.patient_dict['hospital_dir'])

            check_str = check_hospital_files(patient)
        else:
            check_str = 'missing'
        df_check_everything[patient_id]['raw files'] = check_str

        # --- check if there is a report
        if patient.patient_dict['report'] == '':
            df_check_everything[patient_id]['report'] = 'Missing'
        else:
            if not os.path.isfile(os.path.join(patient.dir, patient.patient_dict['report'])):
                df_check_everything[patient_id]['report'] = 'Missing'
            else:
                df_check_everything[patient_id]['report'] = 'Ok'

        # --- check if Bitalino files are here
        if patient.patient_dict['wearable_dir'] == '':
            df_check_everything[patient_id]['bitalino'] = ''
        else:
            if not os.path.isdir(os.path.join(patient.dir, patient.patient_dict['wearable_dir'])):
                df_check_everything[patient_id]['bitalino'] = 'Missing'
            else:
                df_check_everything[patient_id]['bitalino'] = 'Ok'
        df_check_everything[patient_id]['source'] = patient.patient_dict['source']
    df_table = pd.DataFrame(df_check_everything).T
    print('ok')
        
      
    