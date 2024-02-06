# Biosignal class for sensor data processing
#
# Created by: Mariana Abreu
# Created on: 2023-10-17
#

# built-in libraries
import os

# third-party libraries
import numpy as np
import pandas as pd
import tsfel

# local libraries
from preepiseizures.src import Patient, biosignal_processing


class Biosignal():
    """
    Biosignal reading and processing
    
    Attributes
    ----------

    """
    def __init__(self, sensor, device=None, source=None) -> None:
        
        self.dir = dir
        self.sensor = sensor
        self.device = device
        self.source = source
        self.FS = None
        self.cfg_file = tsfel.get_features_by_domain()
        pass

    def get_acc_signals(self, data, patient_class):
        
        """
        ACC signal from data values table 
        ----
        Parameters:
            data (pd.DataFrame): data values table or fraction of it
            device (str): device to extract
        Returns:
            dataframe: dataframe with accelerometer signals
        """
        if 'wearable_device_2' in patient_class.patient_dict.keys(): # has two devices

            if 'EDA' in patient_class.patient_dict['wearable_sensors'].values():
                # wristbit is the first device
                if self.device == 'wristbit':
                    sensor_signal = data.iloc[:,:11][['ACCX', 'ACCY', 'ACCZ']]
                else:
                    sensor_signal = data.iloc[:,11:][['ACCX', 'ACCY', 'ACCZ']]
            else:
                if self.device == 'wristbit':
                    sensor_signal = data.iloc[:,11:][['ACCX', 'ACCY', 'ACCZ']]
                else:
                    sensor_signal = data.iloc[:,:11][['ACCX', 'ACCY', 'ACCZ']]
        else: 
            sensor_signal = data[['ACCX', 'ACCY', 'ACCZ']]
        return sensor_signal
    
    def get_data_hr(self, patient_class, filedir, hosp_class=None) -> pd.DataFrame:
        """
        Get data from file and return hr
        ----
        Parameters:
            filedir: str - Path to file - whole path and ends with .txt
            source: str - 'hospital' or 'wearable'
            fs: int - Sampling frequency
            quality_times: pd.DataFrame - Dataframe with quality times
            hsm_class: Patient.HSMdata - HSM class
        ----
        Returns:
            ecg_hr: pd.DataFrame - Dataframe with hr data
        """
        if self.source not in ['hospital', 'wearable']:
            raise IOError('Source must be either "hospital" or "wearable"')
        
        if self.FS is None:
            raise IOError('Sampling frequency must be defined')
        else:
            fs = self.FS

        if self.source == 'hospital':
            if hosp_class is None:
                raise IOError('Needs the HSM class as input')
            data, start_time = hosp_class.read_file_data(filedir)
            ecg_df = data['ECG2'] - data['ECG1']
            ecg_df.index = pd.date_range(start_time, periods=len(ecg_df), freq=str(1/fs)+'S')
        if self.source == 'wearable':
            patient_class.datafile.process_chunks(filedir, patient_class.patient_dict)
            start_time = patient_class.datafile.start_date
            data = patient_class.datafile.data_values
            time_range = pd.date_range(start_time, periods=len(data), freq=str(1/fs)+'S')
            data.index = time_range
            data = Patient.correct_patient(data, patient_class.id)
            data.index = data['datetime']  
            ecg_df = data['ECG']  

        ecg_hr = biosignal_processing.ecg_processing_to_hr(ecg_df, fs)
        return ecg_hr
    
    def get_ecg_data(self, patient_class, filedir, hosp_class=None) -> pd.DataFrame:
        """
        Get data from file and return ecg data processed
        ----
        Parameters:
            filedir: str - Path to file - whole path and ends with .txt
            source: str - 'hospital' or 'wearable'
            fs: int - Sampling frequency
            quality_times: pd.DataFrame - Dataframe with quality times
            hsm_class: Patient.HSMdata - HSM class
        ----
        Returns:
            ecg_hr: pd.DataFrame - Dataframe with hr data
        """
        if self.source not in ['hospital', 'wearable']:
            raise IOError('Source must be either "hospital" or "wearable"')
        
        if self.FS is None:
            raise IOError('Sampling frequency must be defined')
        else:
            fs = self.FS

        if self.source == 'hospital':
            if hosp_class is None:
                raise IOError('Needs the HSM class as input')
            data, start_time = hosp_class.read_file_data(filedir)
            if len(data) < 1:
                return pd.DataFrame()
            ecg_df = data['ECG2'] - data['ECG1']
            ecg_df.index = pd.date_range(start_time, periods=len(ecg_df), freq=str(1/fs)+'S')
        if self.source == 'wearable':
            patient_class.datafile.process_chunks(filedir, patient_class.patient_dict)
            start_time = patient_class.datafile.start_date
            data = patient_class.datafile.data_values
            time_range = pd.date_range(start_time, periods=len(data), freq=str(1/fs)+'S')
            data.index = time_range
            data.index = data['datetime']  
            ecg_df = data['ECG']  

        return ecg_df
    
    def get_ecg_hospital_data(self, patient_class, filepath='', files_list=[]) -> pd.DataFrame:
        """
        Get all ecg from hospital files and save to parquet files
        Since this can take a while, each file is processed individually and saved to a parquet file indivdually
        The name of the parquet file will be the same as the txt file plus sensor_ecg80.parquet
        The data is filtered and resampled to 80Hz and for each raw ecg file a parquet file is saved
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
            files_list: list - List of files to process - if empty, process all files - files need to end with txt and exist in filepath
        ----
        Returns:
            None
        """
        # define filename where the sensor processed data will be saved
        # if the filepath already exists, get sensor data without any other processing

        # directory to sensor files
        if os.path.isdir(os.path.join(patient_class.dir, 'raw_eeg')):
            hosp_dir = os.path.join(patient_class.dir, 'raw_eeg')
        else:
            hosp_dir = os.path.join(patient_class.dir, patient_class.patient_dict['hospital_dir'])   
        # all_acc = pd.DataFrame()
        i = 0
        # if files_list is empty, process all files
        if files_list == []:
            files_list = sorted(list(filter(lambda file: (file.upper().endswith('.EDF') or file.upper().endswith('.EEG') or file.upper().endswith('.TRC')), os.listdir(hosp_dir))))
        if files_list == []:
            print('No files found')
            return -1
        if patient_class.patient_dict['source'] in ['HEM', 'HEM/Retrospective']:
            hosp_class = Patient.HEMdata(patient_id=patient_class.id, dir=patient_class.dir)
        else:
            hosp_class = Patient.HSMdata(patient_id=patient_class.id, dir=patient_class.dir)
        
        self.FS = int(hosp_class.FS)
        # print(files_list)
        i = 0
        # termination of files is different for each hospital
        # hospital_files_ending = [file for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf', 'trc']]
        for filename in files_list:
            
            filepath = f'data{os.sep}segments{os.sep}{patient_class.id}{os.sep}{filename[:-4]}_{patient_class.id}_{self.sensor}_hospital_ecg80_2024.parquet'
            # get ecg data resampled from hospital
            if os.path.isfile(filepath):
                continue
                # ecg_df = pd.read_parquet(filepath)
            else:
                # find the file with the same name
                print(f"Processing {i}/{len(files_list)}", end='\r')  
                i += 1
                hosp_ecg = self.get_ecg_data(patient_class, os.path.join(hosp_dir, filename), hosp_class=hosp_class)
                if hosp_ecg.empty:
                    print(f'Empty file {filename}')
                    continue
                ecg_df = biosignal_processing.ecg_processing(hosp_ecg, self.FS)
                if not os.path.isdir('data' + os.sep + 'segments' + os.sep + patient_class.id):
                    os.mkdir('data' + os.sep + 'segments' + os.sep + patient_class.id)
                ecg_df.to_parquet(filepath, engine='fastparquet')
        return 1
    
    def calc_ecg_hospital_segments(self, patient_class, filepath='', files_list=[]) -> pd.DataFrame:
        """
        Open parquet files with resampled data and join them together
        Find segmentation times and process each segment individually
        For each segment check quality and normalise between 0 and 1
        Save again into one parquet file with all segments
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
            files_list: list - List of files to process - if empty, process all files - files need to end with txt and exist in filepath
        ----
        Returns:
            acc_df: pd.DataFrame - Dataframe with all acc data

        """
        # define filename where the sensor processed data will be saved
        # if the filepath already exists, get sensor data without any other processing
        window = pd.Timedelta(seconds=30)
        overlap = 0.5
        # directory to sensor files
        if os.path.isfile(f'data{os.sep}segments{os.sep}{patient_class.id}_{self.sensor}_hospital_segments.parquet'):
            print('Patient done!')
            return None


        filedir = os.path.join(f'data{os.sep}segments{os.sep}{patient_class.id}')
        files_list = sorted([file for file in os.listdir(filedir) if (file.endswith('ecg80.parquet')) and (patient_class.id in file)])
        if files_list == []:
            self.get_ecg_hospital_data(patient_class)
        files_list = sorted([file for file in os.listdir(filedir) if (file.endswith('ecg80.parquet')) and (patient_class.id in file)])
        
        all_ecg = pd.concat([pd.read_parquet(os.path.join(filedir, file)) for file in files_list])
        all_ecg = Patient.correct_patient(all_ecg, patient_class.id)
        all_ecg.drop(columns=['datetime'], inplace=True)
        segment_times = pd.date_range(all_ecg.index[0], all_ecg.index[-1], freq=str((window*overlap).total_seconds()) + 'S')
        all_ecg['timestamp'] = all_ecg.index
        all_ecg.reset_index(drop=True, inplace=True)
        all_segments = pd.concat(list(map(lambda x: biosignal_processing.ecg_segment_norm(all_ecg.loc[all_ecg['timestamp'].between(x, x+window)].copy()), segment_times)))
        all_segments.to_parquet(f'data{os.sep}segments{os.sep}{patient_class.id}_{self.sensor}_hospital_segments.parquet', engine='fastparquet')

    def calc_resp_wearable_segments(self, patient_class, filepath='', files_list=[]) -> pd.DataFrame:
        """
        Get all resp from wearable files and save to parquet files
        Since this can take a while, each file is processed individually and saved to a parquet file indivdually
        The name of the parquet file will be the same as the txt file plus sensor_features.parquet
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
            files_list: list - List of files to process - if empty, process all files - files need to end with txt and exist in filepath
        ----
        Returns:
            acc_df: pd.DataFrame - Dataframe with all acc data

        """
        # define filename where the sensor processed data will be saved
        #if filepath == '':
        #    filepath = f'data{os.sep}features{os.sep}{filename[:-4]}_{patient_class.id}_{self.source}_{self.sensor}_features.parquet'
        # if the filepath already exists, get sensor data without any other processing
        if os.path.isfile(filepath):
            # return pd.read_parquet(filepath)
            return None

        if self.FS is None:
            self.FS = int(1000)
        fs = self.FS
        # directory to sensor files
        filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'])
        all_resp = pd.DataFrame()
        i = 0
        # if files_list is empty, process all files
        if files_list == []:
            files_list = [file for file in os.listdir(filedir) if file.startswith('A20')]
        
        if not os.path.isdir('data' + os.sep + 'respiration' + os.sep + patient_class.id):
            os.mkdir('data' + os.sep + 'respiration' + os.sep + patient_class.id)

        for filename in files_list:
            # get data from one file
            print(f"Processing {i}/{len(files_list)}", end='\r')  
            i += 1
            filepath = f'data{os.sep}respiration{os.sep}{patient_class.id}{os.sep}{filename[:-4]}_{patient_class.id}_{self.sensor}_data.parquet'
            if os.path.isfile(filepath):
                resp_data = pd.read_parquet(filepath)
            else:
                patient_class.datafile.process_chunks(filedir + os.sep + filename, patient_class.patient_dict)
                data = patient_class.datafile.data_values
                if data.empty:
                    continue
                # get a timestamp per sample
                time_range = pd.date_range(patient_class.datafile.start_date, periods=len(data), freq=str(1/fs)+'S')
                data.index = time_range            
                if 'PZT' not in data.columns:
                    raise IOError('No PZT column in data')
                resp_data = biosignal_processing.resp_processing(data, fs)
                # data values should have std != 0
                resp_data.to_parquet(filepath, engine='fastparquet')
            
            resp_data['datetime'] = resp_data.index
            all_resp = pd.concat((all_resp, resp_data), ignore_index=True)
        
        all_resp = Patient.correct_patient(all_resp, patient_class.id)

            # all_acc = pd.concat((all_acc, acc_features), ignore_index=True)
        

        #all_acc = Patient.correct_patient(all_acc, patient_class.id)
        # all_acc['timestamp'] = all_acc.index
        all_resp.reset_index(drop=True, inplace=True)
        all_resp['datetime'] = pd.to_datetime(all_resp['datetime'], infer_datetime_format=True)
        all_resp.to_parquet(f'data{os.sep}respiration{os.sep}{patient_class.id}_all_respiration_data.parquet', engine='fastparquet')

        #return all_acc
   

    def calc_acc_wearable_features(self, patient_class, filepath='', files_list=[]) -> pd.DataFrame:
        """
        Get all acc from wearable files and save to parquet files
        Since this can take a while, each file is processed individually and saved to a parquet file indivdually
        The name of the parquet file will be the same as the txt file plus sensor_features.parquet
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
            files_list: list - List of files to process - if empty, process all files - files need to end with txt and exist in filepath
        ----
        Returns:
            acc_df: pd.DataFrame - Dataframe with all acc data

        """
        # define filename where the sensor processed data will be saved
        #if filepath == '':
        #    filepath = f'data{os.sep}features{os.sep}{filename[:-4]}_{patient_class.id}_{self.source}_{self.sensor}_features.parquet'
        # if the filepath already exists, get sensor data without any other processing
        if os.path.isfile(filepath):
            return pd.read_parquet(filepath)

        if self.FS is None:
            self.FS = int(1000)
        fs = self.FS
        # directory to sensor files
        filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'])
        # all_acc = pd.DataFrame()
        i = 0
        # if files_list is empty, process all files
        if files_list == []:
            files_list = os.listdir(filedir)

        for filename in files_list:
            # get data from one file
            print(f"Processing {i}/{len(files_list)}", end='\r')  
            i += 1
            if os.path.isfile(f'data{os.sep}features{os.sep}{filename[:-4]}_{patient_class.id}_{self.sensor}_features.parquet'):
                continue
            patient_class.datafile.process_chunks(filedir + os.sep + filename, patient_class.patient_dict)
            data = patient_class.datafile.data_values
            if data.empty:
                continue
            # get a timestamp per sample
            time_range = pd.date_range(patient_class.datafile.start_date, periods=len(data), freq=str(1/fs)+'S')
            data.index = time_range
            data = self.get_acc_signals(data, patient_class)
            
            acc_data = biosignal_processing.acc_processing(data, fs)
            # data values should have std != 0
            if acc_data[['ACCX', 'ACCY', 'ACCZ']].std().sum() < 0.05:
                acc_features = pd.DataFrame()
            else:
                try:
                    acc_features = self.get_features_from_segments(acc_data)
                except:
                    print(f'Error in processing {filename}')
            
            acc_features.to_parquet(f'data{os.sep}features{os.sep}{filename[:-4]}_{patient_class.id}_{self.sensor}_features.parquet', engine='fastparquet')
            # all_acc = pd.concat((all_acc, acc_features), ignore_index=True)
            

        #all_acc = Patient.correct_patient(all_acc, patient_class.id)
        # all_acc['timestamp'] = all_acc.index
        # all_acc.reset_index(drop=True, inplace=True)
        #all_acc.to_parquet(filepath, engine='fastparquet')

        #return all_acc
    
    def get_features_from_segments(self, data):
        """
        Get segments of data this was optimised for the accelerometer data
        ----
        Parameters:
            data: pd.DataFrame - Dataframe with data where index is datetimeindex
            window: pd.Timedelta - Window size
            overlap: pd.Timedelta - Overlap size
        ----
        Returns:
            timesteps: list - List of timestamps
        
        """
        window = pd.Timedelta(seconds=10)
        overlap = window * (1 - 0.5)
        data['timestamp'] = data.index

        # this line creates a list of timestamps with the desired window and overlap
        timesteps = data.resample(rule= window-overlap, on='timestamp')['timestamp'].min()

        # this line creates a list of dataframes with the data around the timestamps
        acc_segments = [data.loc[data['timestamp'].between(timestamp, timestamp + window)].drop(columns=['timestamp', 'ACCI']) for timestamp in timesteps]
        
        # all segments with less than 10 seconds are removed
        acc_segments_10s = list(filter(lambda x: len(x) > int(window.total_seconds())*self.FS, acc_segments))
        if len(acc_segments_10s) == 0:
            return pd.DataFrame()
        acc_features = self.get_features_acc(acc_segments_10s)

        timestamp_features = list(map(lambda x: x.index[0], acc_segments_10s))
        acc_features['timestamp'] = timestamp_features
        return acc_features
    
    def get_features_acc(self, data):
        """
        Get features from data
        ----
        Parameters:
            data: pd.DataFrame - Dataframe with data
        ----
        Returns:
            features: pd.DataFrame - Dataframe with features
        """
        # get features from each segment
        features = tsfel.time_series_features_extractor(self.cfg_file, data, fs=self.FS, njobs=-1, header_names=['ACCX', 'ACCY', 'ACCZ', 'ACCM'])
        # extract all features
        return features
  
    def get_all_hr_wearable(self, patient_class, filepath='') -> pd.DataFrame:
        """
        Get all hr from wearable files and save to parquet file
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
        ----
        Returns:
            hr_df: pd.DataFrame - Dataframe with all hr data
        """
        self.FS = 1000
        if filepath == '':
            filepath = f'{patient_class.id}_wearable_hr.parquet'

        if os.path.isfile(filepath):
            hr_df = pd.read_parquet(filepath, engine='fastparquet')
            return hr_df
        
        pat_quality = patient_class.get_quality_data()

        # else calculate hr 
        all_hr = pd.DataFrame(columns=['hr'])
        i = 0
        for file in pat_quality['filename'].unique():
            print(f"Processing {i}/{len(pat_quality['filename'].unique())}", end='\r')  
            i += 1
            filedir = os.path.join(patient_class.dir, patient_class.patient_dict['wearable_dir'], file + '.txt')
            wear_hr = self.get_data_hr(patient_class, os.path.join(patient_class.dir, filedir))
            all_hr = pd.concat((all_hr, wear_hr))

        # i don't know why but column 'hr' is always nan while '0' contains the hr values
        if 'hr' in all_hr.columns:
            all_hr.drop(columns=['hr'], inplace=True)
        all_hr.rename(columns={0: 'hr'}, inplace=True)
        all_hr['timestamp'] = all_hr.index
        all_hr.reset_index(drop=True, inplace=True)
        all_hr.to_parquet(f'{patient_class.id}_wearable_hr.parquet', engine='fastparquet')

        print('Success in creating HR df!')
        return all_hr

    def get_all_hr_hospital(self, patient_class, filepath='') -> pd.DataFrame:
        """
        Get all hr from hospital files and save to parquet file
        ----
        Parameters:
            patient_class: Patient - Patient class
            filepath: str - Path to parquet file
        ----
        Returns:
            hr_df: pd.DataFrame - Dataframe with all hr data
        """
        if filepath == '':
            filepath = f'{patient_class.id}_hospital_hr.parquet'
            
        if os.path.isfile(filepath):
            hr_df = pd.read_parquet(filepath, engine='fastparquet')
            return hr_df
        
        hospital_quality = patient_class.get_quality_data(source='hospital')

        if os.path.isdir(os.path.join(patient_class.dir, 'raw_eeg')):
            hosp_dir = os.path.join(patient_class.dir, 'raw_eeg')
        else:
            hosp_dir = os.path.join(patient_class.dir, patient_class.patient_dict['hospital_dir']) 

        all_hr = pd.DataFrame()
        i = 0
        if patient_class.patient_dict['source'] == 'HEM':
            hosp_class = Patient.HEMdata(patient_id=patient_class.id, dir=patient_class.dir)
        else:
            hosp_class = Patient.HSMdata(patient_id=patient_class.id, dir=patient_class.dir)
        
        self.FS = int(hosp_class.FS)

        if hospital_quality.empty:
            print('No hospital quality data available')
            hospital_files = [file.split('.')[0] for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf', 'trc']]
        else:
            hospital_files = hospital_quality['filename'].unique()
        
        # termination of files is different for each hospital
        hospital_files_ending = [file for file in os.listdir(hosp_dir) if file.lower()[-3:] in ['eeg', 'edf', 'trc']]
        for filename in hospital_files:
            # find the file with the same name
            file_with_ending = [file for file in hospital_files_ending if filename in file][0]
            print(f"Processing {i}/{len(hospital_files)}", end='\r')  
            i += 1
            hosp_hr = self.get_data_hr(patient_class, os.path.join(hosp_dir, file_with_ending), hosp_class=hosp_class)
            all_hr = pd.concat([all_hr, hosp_hr])
        if 'hr' in all_hr.columns:
            all_hr.drop(columns=['hr'], inplace=True)
        all_hr.rename(columns={0: 'hr'}, inplace=True)
        all_hr['timestamp'] = all_hr.index
        all_hr.reset_index(drop=True, inplace=True)
        all_hr.to_parquet(f'{patient_class.id}_hospital_hr.parquet', engine='fastparquet')
        print('Success in creating HR df!')

        return all_hr