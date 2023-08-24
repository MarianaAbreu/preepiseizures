#built-in
from datetime import datetime
import json
import os
from types import SimpleNamespace

#external
import numpy as np
import pandas as pd
import biosppy as bp
from neo import MicromedIO
from mne.io import read_raw_edf
from scipy.signal import find_peaks

#local
from .biosignal_analysis import get_ecg_analysis, get_resp_analysis

class Patient():
    """
    Patient class to store patient information
    """
    def __init__(self, patient_id, dir) -> None:
        self.id = patient_id

        self.patient_dict = self.get_patient_dict()
        self.dir = self.get_patient_dir(dir)
        self.seizure_table = pd.DataFrame()
        self.pos_label = 'ecg'
        self.neg_label = 'ECG'

    def get_patient_dir(self, dir):
        """
        Get patient directory from the given directory
        Parameters:
            dir (str): directory to search for patient directory
        Returns:
            str: functional patient directory or error
        """

        if os.path.isdir(dir):
            if self.id in os.listdir(dir):
                return os.path.join(dir, self.id)
            else:
                if os.path.isdir(os.path.join(dir, self.patient_dict['touch'])):
                    return os.path.join(dir, self.patient_dict['touch'])
                else:
                    if os.path.isdir(os.path.join(dir, self.id)):
                        return os.path.join(dir, self.id)
                    raise IOError(f"Patient directory not found in {dir}")
                
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

        seizure_dir = [file for file in os.listdir(self.dir) if ("seizure_label" in file and not file.startswith('._')) ]
        if len(seizure_dir) == 1:
            self.seizure_table = pd.read_csv(os.path.join(self.dir, seizure_dir[0]), index_col=0) 
        else:
            print(seizure_dir)
            raise IOError(f"{len(seizure_dir)} Files found inside directory with key 'seizure_label'")   


class HSMdata():

    def __init__(self, patient_id, dir) -> None:
        self.FS = 1000
        self.dir = dir
        self.id = patient_id

    def read_edf_file_data(filedir):
        # TODO
        hsm_data = read_raw_edf(filedir)
        # get channels that correspond to type (POL Ecg = type ecg)
        channel_list = [hch for hch in hsm_data.ch_names if 'ecg' in hch.lower()]
        # initial datetime
        
        # sampling frequency
        self.FS = hsm_data.info['sfreq']
        # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
        hsm_sig = hsm_data[channel_list]

        return hsm_sig[0], hsm_data.info['meas_date'].replace(tzinfo=None)
        

    def read_eeg_file_data(filedir):
        # TODO
        pass
    


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
        