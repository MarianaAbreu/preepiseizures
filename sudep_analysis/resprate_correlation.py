# Script to calculate the correlation between the respiratory rate and the other features
#
# Author: Mariana Abreu
# Date: 02/02/2024

# import built-in libraries
import os

# import third-party libraries
import pandas as pd

# import local libraries
from preepiseizures.src import Patient, biosignal_processing


if __name__ == '__main__':


    # get all patients
    patients = [file.split('_')[0] for file in os.listdir('data/autoencoders_epilepsy/') if file.endswith('corr_points_1s.parquet')]

    for patient in sorted(patients):
        print('Processing patient', patient)
        # get patient metadata ----------
        patient_info = Patient.patient_class(patient)
        corr_data = pd.read_parquet(os.path.join('data/autoencoders_epilepsy', patient + '_corr_points_1s.parquet'))

        data = pd.read_parquet('data/respiration/{patient}_all_respiration_data.parquet'.format(patient=patient))
        
        if patient in ['BLIW', 'YWJN']:
            data['datetime'] += pd.Timedelta(patient_info.patient_dict['temporal_shift']) 
        
        resp_rate = pd.DataFrame()
        for i in range(len(corr_data)):
            print(f'{i}/{len(corr_data)}', end='\r')
            start_time = corr_data.iloc[i].name
            data_segment = data.loc[data['datetime'].between(start_time, start_time + pd.Timedelta(minutes=1))]['RESP'].copy()
            rate = biosignal_processing.resp_rate(data_segment, sampling_rate=80, overlap=60, window_size=60)
            resp_rate = pd.concat([resp_rate, pd.DataFrame({'rate': rate[0], 'datetime': start_time})], ignore_index=True)

        corr_data = corr_data.merge(resp_rate, left_index=True, right_on='datetime')
        corr_data.to_parquet('data/autoencoders_epilepsy/{patient}_corr_points_1s_resp_rate.parquet'.format(patient=patient))
