#built-in
from datetime import datetime
import json
import os
import string
from types import SimpleNamespace

#external
import numpy as np
import pandas as pd
import biosppy as bp
from neo import MicromedIO
from mne.io import read_raw_edf, read_raw_nihon
from scipy.signal import find_peaks

from preepiseizures.src import OpenFileProcessor, biosignal_plot, biosignal_processing

#local
from .biosignal_analysis import get_ecg_analysis, get_resp_analysis

class Patient():
    """
    Patient class to store patient information
    """
    def __init__(self, patient_id, main_dir) -> None:

        self.id = patient_id
        self.patient_dict = self.get_patient_dict()
        self.dir = self.get_patient_dir(main_dir)
        self.datafile = OpenFileProcessor.OpenFileProcessor(mode='EPIBOX')
        self.seizure_table = pd.DataFrame()
        self.pos_label = 'ecg'
        self.neg_label = 'ECG'

    def get_patient_dir(self, main_dir):
        """
        Get patient directory from the given directory
        Parameters:
            dir (str): directory to search for patient directory
        Returns:
            str: functional patient directory or error
        """

        if os.path.isdir(main_dir):
            if self.id in os.listdir(main_dir):
                return os.path.join(main_dir, self.id)
            else:
                if os.path.isdir(os.path.join(main_dir, self.patient_dict['touch'])):
                    return os.path.join(main_dir, self.patient_dict['touch'])
                else:
                    if os.path.isdir(os.path.join(main_dir, self.id)):
                        return os.path.join(main_dir, self.id)
                    raise IOError(f"Patient {self.id} directory not found in {main_dir}")
                
    def get_patient_dict(self):
        """
        Get patient dictionary
        Parameters:
            filesdir (str): directory where the files are located
        Returns:
            dict: dictionary with patient information
        """

        with open("patient_info.json") as fh:
            all_patient_dict = json.load(fh)

        if self.id in all_patient_dict.keys():
            patient_dict = all_patient_dict[self.id]
        else:
            print(f'{self.id} not found in patient_info.json')
            patient_dict = {}
        # hospital directory:
        
        return patient_dict

    def get_seizure_annotations(self):

        self.seizure_table = pd.DataFrame()

        if self.patient_dict['source'] == 'HSM':

            excel_file = 'Patients_HSM_.xlsx'
        
        else:
            excel_file = 'Pat_HEM.xlsx'
        try:
            seizure_table = pd.read_excel(os.path.join('/Volumes', 'T7 Touch', 'PreEpiSeizures', 'Patients_'+self.patient_dict['source'], excel_file), sheet_name=self.patient_dict['touch'])
            self.seizure_table = seizure_table[seizure_table['Crises'].notna()].copy()
        except:
            if self.patient_dict['wearable_dir'] == '':
                print(f"Seizure label file not found for patient {self.id} but also no wearable data")
            else:
                print(f"Seizure label file not found for patient {self.id}")
        if (self.seizure_table.empty or self.seizure_table['Crises'].iloc[0] == 0):
            print(f"No seizure annotations found for patient {self.id}")
            return None
        self.seizure_table.loc[:, 'Data'] = [datetime.strptime(date, '%d-%m-%Y') if type(date) is str else date for date in self.seizure_table['Data']]
        self.seizure_table.loc[:, 'Timestamp'] = [datetime.combine(self.seizure_table.iloc[sidx]['Data'], 
                                                            self.seizure_table.iloc[sidx]['Hora Clínica']) for 
                                                            sidx in range(len(self.seizure_table))]
        # else: 
        #     try: 
        #         seizure_table = pd.read_excel('/Volumes/T7 Touch/PreEpiSeizures/Patients_HSM/Pat_HEM_.xlsx', sheet_name=self.patient_dict['touch'])
        #         self.seizure_table = seizure_table[seizure_table['Crises'].notna()].copy()
        #     except:
        #         if self.patient_dict['wearable_dir'] == '':
        #             print(f"Seizure label file not found for patient {self.id} but also no wearable data")
        #         else:
        #             print(f"Seizure label file not found for patient {self.id}")

        #     seizure_dir = [file for file in os.listdir(self.dir) if ("seizure_label" in file and not file.startswith('._')) ]
        #     if len(seizure_dir) == 1:
        #         self.seizure_table = pd.read_csv(os.path.join(self.dir, seizure_dir[0]), index_col=0) 
        #     if self.seizure_table.empty:
        #         print(f"No seizure annotations found for patient {self.id}")
        #         return None
        #     self.seizure_table.loc[:, 'Timestamp'] = [datetime.strptime(date, '%d-%m-%Y\n%H:%M:%S') for date in self.seizure_table['Date']]
        if 'Timestamp' not in self.seizure_table.columns:
            print(f"No seizure annotations found for patient {self.id}")
            return None

    def get_all_acc_data(self, sensor, source, savedir=''):
        """
        Extract all data from one sensor modality of one patient
        ---
        Parameters:
            sensor (str): sensor to extract
            source (str): source of the data (wearable or hospital)
            savedir (str): directory to save the data
        Returns:
            dataframe: dataframe with sensor data. This dataframe contains one column timestamp (or datetime) and one column per sensor
        """

        if savedir == '':
            savedir = os.path.join('data', f'{self.id}_{sensor}_data.parquet')
        
        if os.path.isfile(savedir):
            return pd.read_parquet(savedir)
        
        filepath = os.path.join(self.dir, self.patient_dict['wearable_dir'])
        new_df = pd.DataFrame()

        for filename in os.listdir(filepath):
            self.datafile.process_chunks(filepath + os.sep + filename, self.patient_dict)
            filename_data = self.datafile.data_values.copy()

        if sensor == 'acc':
            acc_df = self.get_acc_signals(filename_data, 'chestbit')

        return new_df    

    def get_wearable_data_baseline(self, sensor='acc', sleep_only=True, hq=True):
        """
        Get wearable data away from seizures
        ---
        Parameters:
            sensor (str): sensor to extract
        Returns:
            dataframe: dataframe with wearable data
        """
        if self.seizure_table.empty:
            self.get_seizure_annotations()
        quality = self.get_quality_data()
        # remove 2 hours surrounding the seizures from the quality data
        for sidx in range(len(self.seizure_table)):
            start_interval = self.seizure_table.iloc[sidx]['Timestamp'] - pd.Timedelta(hours=1)
            end_interval = self.seizure_table.iloc[sidx]['Timestamp'] + pd.Timedelta(hours=1)
            quality = quality.loc[~quality['datetime'].between(start_interval, end_interval)]  
        if hq:
            quality = quality.loc[quality['quality']> 0.5]
        if sleep_only:
            start_day = quality['datetime'].iloc[0].date()
            end_day = quality['datetime'].iloc[-1].date()
            # get sleep nights
            sleep_nights = []
            for day in pd.date_range(start_day, end_day):
                start_sleep = day + pd.Timedelta(hours=23)
                end_sleep = day + pd.Timedelta(hours=31)
                sleep_nights.append(quality.loc[quality['datetime'].between(start_sleep, end_sleep)])
            quality = pd.concat(sleep_nights, ignore_index=True)
        files = quality['filename'].unique()

        all_df = pd.DataFrame()
        for filename in files:
            start_interval = quality.loc[quality['filename']==filename]['datetime'].iloc[0]
            end_interval = quality.loc[quality['filename']==filename]['datetime'].iloc[-1]
            self.datafile.process_chunks(os.path.join(self.dir, self.patient_dict['wearable_dir'], 
                                                      filename + '.txt'), self.patient_dict)
            # get time range
            time_range = pd.date_range(self.datafile.start_date, periods=len(self.datafile.data_values), 
                                       freq=str(1/self.datafile.fs)+'S')
            # get data in time range
            filename_data = self.datafile.data_values.copy()
            filename_data.index = time_range
            filename_data = correct_patient(filename_data, self.id)
            filename_data.index = filename_data['datetime']
            filename_data = filename_data.loc[start_interval: end_interval]
            if filename_data.empty:
                continue

            if sensor == 'acc':
                acc_df = self.get_acc_signals(filename_data, 'chestbit')
                acc_df.loc[:, 'datetime'] = filename_data.index
                if len(acc_df) < 1000:
                    continue
                acc_df = biosignal_processing.acc_processing(acc_df, self.datafile.fs)
                # processing
                acc_df.loc[:, 'filename'] = filename
                all_df = pd.concat([all_df, acc_df], ignore_index=True)
            elif sensor == 'ecg':
                ecg_df = filename_data['ECG']
                ecg_df = biosignal_processing.ecg_processing(ecg_df, self.datafile.fs)
                ecg_df.loc[:, 'datetime'] = ecg_df.index
                all_df = pd.concat([all_df, ecg_df], ignore_index=True)


        return all_df 
    
    def get_hospital_data_around_seizures(self, sensor='ecg'):
        """
        Get hospital data
        Get all available sensors
        """
        if self.seizure_table.empty:
            self.get_seizure_annotations()
        if self.seizure_table.empty:
            print(f"No seizure annotations found for patient {self.id}")
            return None
        # find files with the same date as the seizure
        hospital_data = pd.DataFrame()
        hsm_class = HSMdata(patient_id=self.id, dir=self.dir)
        if os.path.isdir(os.path.join(self.dir, 'raw_eeg')):
            hosp_dir = os.path.join(self.dir, 'raw_eeg')
        else:
            hosp_dir = os.path.join(self.dir, self.patient_dict['hospital_dir'])    
        datetime_dict = hsm_class.get_file_start_date(hosp_dir)
        datetime_table = pd.DataFrame(datetime_dict.values(), index=datetime_dict.keys(), columns=['filename'])
        datetime_table['datetime'] = datetime_table.index
        for sidx in range(len(self.seizure_table)):
            seiz_time = self.seizure_table.iloc[sidx]['Timestamp']
            all_df = pd.DataFrame()
            # files have 2 hours
            files = datetime_table.loc[datetime_table['datetime'].between(seiz_time - pd.Timedelta(hours=2), 
                                                                          seiz_time + pd.Timedelta(hours=2))]['filename'].unique()
            for filename in files:
                seizure_df = pd.DataFrame()
                # get data
                data, start_date = hsm_class.read_file_data(filename)
                if len(data) < 1:
                    continue           
                # get time range
                time_range = pd.date_range(start_date, periods=len(data), freq=str(1/hsm_class.FS)+'S')
                # get data in time range
                filename_data = data.copy()
                filename_data.index = time_range
                filename_data = filename_data.loc[seiz_time - pd.Timedelta(minutes=30): seiz_time + pd.Timedelta(minutes=30)]
                if filename_data.empty:
                    continue
                # seizure_df = pd.concat([seizure_df, filename_data], ignore_index=True)
                # processing
                if sensor == 'ecg':
                    ecg_df = filename_data['ECG2'] - filename_data['ECG1']
                    ecg_df = biosignal_processing.ecg_processing(ecg_df, int(hsm_class.FS))
                    seizure_df[ecg_df.columns] = ecg_df

                seizure_df.loc[:, 'filename'] = filename
                seizure_df.loc[:, 'seizure'] = sidx
                seizure_df.loc[:, 'datetime'] = filename_data.index
                all_df = pd.concat([all_df, seizure_df], ignore_index=True)
            hospital_data = pd.concat([hospital_data, all_df], ignore_index=True)
        hospital_data['patient'] = self.id
        # self.datafile.process_chunks(os.path.join(self.dir, self.patient_dict['wearable_dir'], filename), self.patient_dict)
        
        return hospital_data

    def get_wearable_data_around_seizures(self, sensor='acc'):
        """
        Get wearable data
        Get all available sensors
        """
        if self.seizure_table.empty:
            self.get_seizure_annotations()
        if self.seizure_table.empty:
            print(f"No seizure annotations found for patient {self.id}")
            return None
        # find files with the same date as the seizure
        quality = self.get_quality_data()  
        
        data = pd.DataFrame()
        for sidx in range(len(self.seizure_table)):
            
            all_df = pd.DataFrame()
            start_interval = self.seizure_table.iloc[sidx]['Timestamp'] - pd.Timedelta(minutes=30)
            end_interval = self.seizure_table.iloc[sidx]['Timestamp'] + pd.Timedelta(minutes=30)
            files = quality.loc[quality['datetime'].between(start_interval, end_interval)]['filename'].unique()
            for filename in files:
                seizure_df = pd.DataFrame()
                # get data
                self.datafile.process_chunks(os.path.join(self.dir, self.patient_dict['wearable_dir'], filename + '.txt'), self.patient_dict)
                # get time range
                time_range = pd.date_range(self.datafile.start_date, periods=len(self.datafile.data_values), freq=str(1/self.datafile.fs)+'S')
                # get data in time range

                filename_data = self.datafile.data_values.copy()
                filename_data.index = time_range
                filename_data = correct_patient(filename_data, self.id)
                filename_data.index = filename_data['datetime']
                filename_data = filename_data.loc[start_interval: end_interval]
                if filename_data.empty:
                    continue
                filename_data.loc[:, 'seizure'] = sidx
                filename_data.loc[:, 'filename'] = filename
                filename_data.loc[:, 'datetime'] = filename_data.index 
                # seizure_df = pd.concat([seizure_df, filename_data], ignore_index=True)
                # processing
                if sensor == 'resp':
                    # TODO
                    resp_df = filename_data['PZT'].copy()
                    resp_df = biosignal_processing.resp_processing(resp_df, self.datafile.fs)
                    seizure_df[resp_df.columns] = resp_df
                
                if sensor == 'acc':
                    for device in ['wristbit', 'chestbit']:
                        acc_df = self.get_acc_signals(filename_data, device)
                        acc_df['datetime'] = filename_data['datetime']
                        acc_df = biosignal_processing.acc_processing(acc_df, self.datafile.fs)
                        seizure_df[acc_df.columns] = acc_df
                        seizure_df.rename(columns={col: col + '_' + device for col in acc_df.columns}, inplace=True)

                if sensor == 'ecg':
                    ecg_df = filename_data['ECG']
                    ecg_df = biosignal_processing.ecg_processing(ecg_df, self.datafile.fs)
                    seizure_df[ecg_df.columns] = ecg_df
                seizure_df['datetime'] = ecg_df.index
                seizure_df['filename'] = filename
                seizure_df['seizure'] = sidx
                all_df = pd.concat([all_df, seizure_df], ignore_index=True)
            if not all_df.empty:
                if sensor == 'acc':
                    biosignal_plot.plot_acc_seizure(all_df, self.seizure_table.iloc[sidx], plotname = sensor+'_seizure_' + str(sidx) + '_' + self.id)
            
            data = pd.concat([data, all_df], ignore_index=True)
        # all df has columns: 
        # wrist_accx, wrist_accy, wrist_accz, chest_accx, chest_accy, chest_accz, 
        # wrist_magnitude, chest_magnitude,
        # wrist_activity_index, chest_activity_index, datetime, filename, seizure
        data['patient'] = self.id
        # self.datafile.process_chunks(os.path.join(self.dir, self.patient_dict['wearable_dir'], filename), self.patient_dict)
        
        return data
    
    def get_quality_data(self, sensor='ecg', source='wearable'):
        """
        Get quality data. Get ECG since it is the signal with the smallest rate (1 sample per 10 seconds)

        """
        if source == 'wearable':
            filename = f"{self.id}_wearable_quality_{sensor.lower()}_incomplete.parquet"
        else:
            filename = f"{self.id}_{source}_quality_incomplete.parquet"
        if os.path.isfile(os.path.join('data', 'quality', filename)):
            quality_df = pd.read_parquet(os.path.join('data', 'quality', filename))
        else:
            # raise FileNotFoundError('Quality file not found')
            return pd.DataFrame()
        # quality_df = correct_patient(quality_df, self.id)
        return quality_df


def patient_class(patient_one):
    """
    Get patient class
    Parameters:
        patient_one (str): patient id
    Returns:
        Patient class
    """

    patient_dict = json.load(open('patient_info.json'))
    folder_dir = ''
    if patient_dict[patient_one]['source'] == 'HSM':
        folder_dir = f"/Volumes/My Passport/Patients_{patient_dict[patient_one]['source']}"
        if not os.path.isdir(os.path.join(folder_dir, patient_one)):
            # print(f'Patient {patient_one} not in folder')
            Warning(f'Patient {patient_one} not in folder')
    else:
        folder_dir = f"/Volumes/T7 Touch/PreEpiSeizures/Patients_{patient_dict[patient_one]['source']}"

    if patient_dict[patient_one]['wearable_dir'] == '':
        #raise FileNotFoundError(f'Patient {patient_one} has no wearable data')
        return None
    patient = Patient(patient_one, folder_dir)
    return patient


def annotation_time_correction(df, dir_):
    """
    Correct time based on the file annotations.txt
    Return correct Dataframe
    """
    annotations = pd.read_csv(dir_, header=None, sep='  ', engine='python')
    # true timestamp 
    true_time = datetime.strptime(annotations.iloc[0][4], '%Y-%m-%d %H:%M:%S.%f')
    # corresponding timestamp saved in the Bitalino files
    unsure_time = datetime.strptime(annotations.iloc[0][6], '%Y-%m-%d %H:%M:%S.%f')
    # correct the timestamps based on the lag between the two timestamps
    df['datetime'] += (true_time - unsure_time)
    return df

def correct_patient(df, patient, patient_dict=json.load(open('patient_info.json'))):
    """ 
    AGGA patient has one file that does not belong to the same date as the others.
    This function removes the rows related to that file.
    ---
    Parameters:
        df: pd.DataFrame
            The dataframe with the quality data of the patient
    ---
    Returns:
        df: pd.DataFrame
            The dataframe with the quality data of the patient without the rows related to the wrong file
    """
    if 'datetime' not in df.columns:
        df['datetime'] = df.index
    df['datetime'] = df['datetime'].astype('datetime64[ns]')
    dir_ = f"/Volumes/My Passport/Patients_HSM/{patient}/{patient_dict[patient]['wearable_dir']}/annotations.txt"
    if os.path.exists(dir_):
        df = annotation_time_correction(df, dir_)
    if patient in ['BLIW', 'YWJN', 'YIVL']:
        temporal_shift = patient_dict[patient]['temporal_shift']
        df['datetime'] += pd.Timedelta(temporal_shift)
        return df
    elif patient == 'QFRK':
        start_date = datetime(2019,7,7,10,4,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'OQQA':
        start_date = datetime(2019,12,10,10,20,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'UIJU':
        start_date = datetime(2019,7,30,10,20,10,321000)
        delta = df['datetime'] - df['datetime'].iloc[0]
        df['datetime'] = start_date + delta
        return df
    elif patient == 'OXDN':
        df_wrong = df.where(df['datetime'].diff() > pd.Timedelta(days=1)).dropna().copy()
        if df_wrong.empty:
            timestamp_wrong = 0
        else:
            timestamp_wrong = df_wrong.index[0]
        return df.iloc[timestamp_wrong:]
    elif patient == 'AGGA':
        timestamp_wrong = df.where(df['datetime'].diff() > pd.Timedelta(days=1)).dropna()
        if timestamp_wrong.empty:
            return df
        else:
            timestamp_wrong = timestamp_wrong.index[0]
            return df.iloc[:timestamp_wrong]
    elif patient == 'VNVW':
        timestamp_wrong_df = df[df['datetime'].diff() > pd.Timedelta(days=1)]
        if timestamp_wrong_df.empty:
            return df
        else:
            timestamp_wrong = timestamp_wrong_df.index[0]
        lag = df.iloc[timestamp_wrong]['datetime'].date()-df.iloc[timestamp_wrong-1]['datetime'].date()
        df.loc[:timestamp_wrong,'datetime'] = df.iloc[:timestamp_wrong]['datetime'] + lag
        return df
    else:
        pass
    return df

class HSMdata():

    def __init__(self, patient_id, dir) -> None:
        self.FS = 1000
        self.dir = dir
        self.id = patient_id

    def read_file_data(self, filedir):
        """
        Read edf file data
        Parameters:
            filedir (str): directory where the file is located and the filename
        Returns:
            array: array with data
            datetime: start date of the file
            int: sampling frequency
        """
        if 'edf' in filedir:
            # open files ending with edf
            try:
                hsm_data = read_raw_edf(filedir, encoding='latin1')
            except Exception as e:
                print(e)
                return [], None
        else:
            # open files ending with EEG
            try:
                hsm_data = read_raw_nihon(filedir)
            except:
                return [], None
        # get channels that correspond to type (POL Ecg = type ecg)
        channel_list = [hch for hch in hsm_data.ch_names if 'ecg' in hch.lower()]
        if len(channel_list) < 1:
            channel_list = [hch for hch in hsm_data.ch_names if hch.lower() in ['a1', 'a2']]
        if len(channel_list) < 1:
            raise ValueError('No ECG channel found')
        # initial datetime
        # sampling frequency
        self.FS = hsm_data.info['sfreq']
        # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
        hsm_sig = hsm_data[channel_list]
        data = pd.DataFrame(hsm_sig[0].T, columns=['ECG1', 'ECG2'])    
        return data, hsm_data.info['meas_date'].replace(tzinfo=None)   

    
    def get_file_start_date(self, filesdir):
        """
        Get start date of the file
        Parameters:
            source (str): source of the file
            filesdir (str): directory where the files are located
        Returns:
            datetime dict: start date of the file
        """
        # APROXIMATED METHOD TO SAVE TIME
        # first file 
        if 'raw_eeg' in filesdir:
            ending = 'eeg'
        else:
            ending = 'edf'
        files = [os.path.join(filesdir, file) for file in sorted(os.listdir(filesdir)) if (file.lower().endswith(ending) and (not file.startswith('._') and os.path.isfile(os.path.join(filesdir, file))))]
        curr_date = self.read_hsm_file_date(files[0])
        datetime_dict = {curr_date: files[0]}
        # get only file key from files dir of the first and last files
        first_file = files[0].split('/')[-1].split('.')[0]
        last_file = files[-1].split('/')[-1].split('.')[0]
        # list of all file keys by order
        file_range_list = self.file_range(first_file, last_file)

        i = 1
        j = 1

        while len(datetime_dict) < len(files):
            next_file = files[i].split('/')[-1].split('.')[0]
            if next_file == file_range_list[j]:
                curr_date += pd.Timedelta(hours=2)
                datetime_dict.update({curr_date: files[i]})
                i += 1

            else:
                # increase curr date without writing any file
                curr_date += pd.Timedelta(hours=2)
            j += 1
                
            
        return datetime_dict
      
    def read_hsm_file_date(self, filedir):
        """
        Read trc file date
        Parameters:
            filedir (str): file directory
        Returns:
            datetime: datetime of the file
        """
        if filedir.endswith('edf'):
            try:
                # TODO
                data = read_raw_edf(filedir, preload=False)
                date = data.info['meas_date'].replace(tzinfo=None)
            except:
                date = None
        else:
            # if file endswith EEG
            data = read_raw_nihon(filedir, preload=False)
            date = data.info['meas_date'].replace(tzinfo=None)

        return date
    

    def file_range(self, first_file, last_file):
        """
        Get file range
        Parameters:
            first_file (str): first file without ending ".edf", ex: 'FA7779ED'
            last_file (str): last file without ending ".edf", ex: 'FA7779ED'
        Returns:
            list: list of files in the range
        """

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



class HEMdata():

    def __init__(self, patient_id, dir) -> None:
        self.FS = 256
        self.dir = dir
        self.id = patient_id
  

    def get_file_start_date(self, source, filesdir):
        """
        Get start date of the file
        Parameters:
            source (str): source of the file
            filesdir (str): directory where the files are located
        Returns:
            datetime dict: start date of the file
        """

        datetime_dict = {}
        if source == "HEM":
            files = [os.path.join(filesdir, file) for file in os.listdir(filesdir) if (not file.startswith('._') and os.path.isfile(os.path.join(filesdir, file)))]
            datetime_dict = dict(zip(map(self.read_trc_file_date, files), files))
        return datetime_dict
    

    def read_file_data(self, filedir):
        """
        Read trc file data
        Parameters:
            filedir (str): file directory
        Returns:
            array: array with data
            datetime: start date of the file
            int: sampling frequency
        """
        if 'trc' in filedir.lower():
            # open files ending with edf
            seg_micromed = MicromedIO(filedir)

        else:
            print('filetype unknown')
            return pd.DataFrame(), None

        # get channels that correspond to type (POL Ecg = type ecg)
        hem_data = seg_micromed.read_segment()
        hem_sig = hem_data.analogsignals[0]
        ch_list = seg_micromed.header['signal_channels']['name']
        find_idx = [hch for hch in range(len(ch_list)) if 'ecg' in ch_list[hch].lower()]
        if len(find_idx) != 2:
            print('ECG channels in trc file are not 2')
            if self.id == 'WVKA':
                find_idx = [33, 34]     
            else:
                print('Solve')   
        data = pd.DataFrame(np.array(hem_sig[:, find_idx]), columns=['ECG1', 'ECG2'])   

        return data, hem_data.rec_datetime
        


    def get_file_df(self, filesdir, filetype):
        """
        Get file dataframe
        Parameters:
            filesdir (list): list of all directories to extract
        Returns:
            dataframe: dataframe with file information
        """
        if filetype.lower() == 'parquet':
            return pd.concat([pd.read_parquet(filedir) for filedir in filesdir], ignore_index=True)
        
        elif filetype.lower() == 'trc':
            df = pd.concat([self.read_trc_file_data(filedir) for filedir in filesdir], ignore_index=True)
            start_date = self.read_trc_file_date(filesdir[0])
            df.index = pd.date_range(start_date, periods=len(df), freq=str(1/self.FS)+'S')
            return df 
        
    def get_seizure_info(self, df, seizure_date, sidx, moment):
        """
        Get seizure information
        Parameters:
            seizure_start (int): seizure start index
            keys_info (list): list of keys to extract
            sidx (int): seizure index
            moment (str): moment of the seizure
        Returns:
            dict: dictionary with seizure information
        """
        if moment == 'preictal':
            start = -pd.Timedelta(minutes=30)
            end = -pd.Timedelta(minutes=5)
        elif moment == 'ictal':
            start = -pd.Timedelta(seconds=10)
            end = pd.Timedelta(minutes=2)
        elif moment == 'posictal':
            start = pd.Timedelta(minutes=5)
            end = pd.Timedelta(minutes=30)
        else:
            raise ValueError(f"Invalid moment {moment}")
        try:        
            sz_df = df.loc[seizure_date + start: seizure_date + end]
        except Exception as e:
            print(e)
            return {}
        if sz_df.shape[0] == 0:
            return {}
        self.define_ecg_labels(sz_df)
        sz_info = get_ecg_analysis(sz_df, self.pos_label, self.neg_label, self.FS)
        sz_info.update(get_resp_analysis(sz_df, self.pos_label, self.neg_label, self.FS))
        sz_info['seizure'] = sidx
        sz_info['seizure_date'] = seizure_date
        sz_info['closest_date'] = sz_df.index[np.argmin(np.abs(sz_df.index - seizure_date))]
        sz_info['duration'] = (sz_df.index[-1] - sz_df.index[0]).total_seconds()
        sz_info['moment'] = moment
        return sz_info
    
    def get_all_seizures_df(self, list_dates):
        """
        Get all seizures dataframe
        Parameters:
            list_dates (dict): dictionary with all dates and file names
        Returns:
            dataframe: dataframe with all seizures information
        """
        
        all_df = pd.DataFrame()
        for sidx in range(len(self.seizure_table)):
            seizure_df = pd.DataFrame()
            seizure_date = datetime.strptime(self.seizure_table.iloc[sidx]['Date'], '%d-%m-%Y\n%H:%M:%S')
            start = seizure_date - pd.Timedelta(minutes=40)
            end = seizure_date + pd.Timedelta(minutes=40)
            seizure_files = [list_dates[date] for date in list_dates.keys() if start <= date <= end]
            if len(seizure_files) == 0:
                continue
            seizure_df = self.get_file_df([file for file in seizure_files], 'trc')
            seizure_df['seizure'] = sidx
            all_df = pd.concat([all_df, seizure_df])
        return all_df
    
    def define_ecg_labels(self, signal_df):

        print('todo')

    def read_trc_file_date(filedir):
        """
        Read trc file date
        Parameters:
            filedir (str): file directory
        Returns:
            datetime: datetime of the file
        """
        seg_micromed = MicromedIO(filedir)
        hem_data = seg_micromed.read_segment(lazy=True)
        return hem_data.rec_datetime

    def read_trc_file_data(filedir):
        """
        Read trc file data
        Parameters:
            filedir (str): file directory
        Returns:
            dataframe: dataframe with file raw data
        """
        seg_micromed = MicromedIO(filedir)
        hem_data = seg_micromed.read_segment()
        hem_signal = hem_data.analogsignals[0]
        ch_list = seg_micromed.header['signal_channels']['name']
        find_idx = [hch for hch in range(len(ch_list)) if 'ecg' in ch_list[hch].lower()]
        return pd.DataFrame(np.array(hem_signal[:, find_idx]), columns=[ch_list[find_idx]])
        