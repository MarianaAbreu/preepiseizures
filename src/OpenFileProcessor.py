
#built-in
import ast
from datetime import datetime
import os
import re
import time

#external
import numpy as np
import pandas as pd
from neo import MicromedIO
from mne.io import read_raw_edf, read_raw_nihon



class OpenFileProcessor:
     """
     This class is used to process the data from a given file.
     It is designed to deal with large files more efficiently.
     """
     CHUNK_SIZE = 1000000  # configure this variable depending on your machine's hardware configuration

     def __init__(self, mode):
          """
          Initialize the class
          ---
          Parameters:
          mode: str 
               The mode of acquisition of the file
          ---
          header_str: str
               The header of the file
          data_values: pd.DataFrame
               The data values of the file
          fs: float
               The sampling frequency of the file
          bit_array: array
               The array of the bit resolution of each channel 
          ---
          """
          self.mode = mode
          self.SEP = ',' if mode.upper() == 'SCIENTISST' else '\t' 
          self.header_str = []
          self.columns = []
          self.data_values = []
          self.fs = 0
          self.bit_array = []
          self.start_date = None

     # def process_lines(self, data, eof, file_name):
     #      """
     #      This is a callback method called for each line of the file.
     #      ---
     #      Parameters:
     #           data: str
     #                The data of the line     
     #           eof: bool
     #                True if end of file reached, False otherwise
     #           file_name: str
     #                The name of the file
     #      ---
     #      """

     #      # check if end of file reached
     #      if not eof:
     #           # process data, data is one single line of the file
     #           self.data_values.append((self.data_values, data), ignore_index=True)
     #      else:
     #           # end of file reached
     #           pass

     def process_header(self):

          if self.mode.upper() in ['EPIBOX', 'BITALINO']:
               try:
                    header = ast.literal_eval(self.header_str[1][24:-2])
                    self.columns = header['column']
               except:
                    # device is the second of the list
                    header = ast.literal_eval(self.header_str[1][2:])
                    self.columns = header[list(header.keys())[0]]['column'] + header[list(header.keys())[1]]['column']

                    header = header[list(header.keys())[1]]
               self.fs = int(header['sampling rate'])
               self.bit_array = header['resolution']
               time_key = [key for key in header if 'time' in key][0]
               date_key = [key for key in header if 'date' in key][0]
               try:
                    self.start_date = datetime.strptime(header[date_key] + ' ' + header[time_key], '"%Y-%m-%d" "%H:%M:%S.%f"')
               except:
                    try:
                         self.start_date = datetime.strptime(header[date_key] + ' ' + header[time_key], '%Y-%m-%d %H:%M:%S.%f')
                    except:
                         self.start_date = datetime.strptime(header[date_key] + ' ' + header[time_key], '%Y-%m-%d %H:%M:%S.')


               print(self.start_date)
               return

          elif self.mode.upper() == 'SCIENTISST':
               try:
                    self.columns = self.header_str[1].split('"')[1].split() # some scientisst files
               except:
                    try: 
                         self.columns = self.header_str[1].strip().split(',') # other scientisst files
                    except:
                         print('Problem on extracting columns in Scientisst')

               # sampling frequency 
               fs_str = [hstr for hstr in self.header_str[0].split(',') if 'Hz' in hstr][0] 
               self.fs = int(re.findall(r'\d+', fs_str)[0])
               # start date
               time_str = [tstr for tstr in self.header_str[0].split(',') if 'ISO' in tstr][0]
               self.start_date = datetime.fromisoformat(time_str.split('"')[3][:-1])
               self.bit_array = np.repeat(12, len(self.columns))
               return
          else:
               raise IOError(f'Mode {self.mode} not supported')

     
     def process_chunks(self, filename, patient_dict=None):

          chunksize = 10 ** 6
          modeSkip = 3 if self.mode.upper() in ['BITALINO', 'EPIBOX'] else 2 if self.mode.upper() == 'SCIENTISST' else 0 # raise IOError(f'Mode {self.mode} not supported')
          size = os.path.getsize(filename)
          if size < 2000:
               self.data_values = pd.DataFrame()
               return
          elif size < 10*chunksize:
               self.data_values = [pd.read_csv(filename, header=None, skiprows=modeSkip, sep=self.SEP)]
               
          else:
               try:
                    with pd.read_csv(filename, chunksize=chunksize, header=None, skiprows=modeSkip, sep=self.SEP) as reader:
                         for chunk in reader:
                              self.data_values += [chunk]
               except:
                    self.data_values = [pd.read_csv(filename, header=None, skiprows=modeSkip, sep=self.SEP)]
          self.header_str = []

          with open(filename) as fh:
               for line in fh:
                    if '#' in line:
                         self.header_str += [line]
                    else:
                         break
          self.process_header()
          if patient_dict:
               cols = {} # dict(zip(range(len(self.columns)), patient_dict['wearable_sensors'].values()))
               for cc, col in enumerate(self.columns):
                    if cc < 8: # len(self.columns)//2:
                         if col in patient_dict['wearable_sensors'].keys():
                              cols[cc] = patient_dict['wearable_sensors'][col]
                         else:
                              cols[cc] = ''
                    else:
                         if col in patient_dict['wearable_sensors_2'].keys():
                              cols[cc] = patient_dict['wearable_sensors_2'][col]
                         else:
                              cols[cc] = ''
          else:
               cols = dict(zip(range(len(self.columns)), self.columns))
          self.data_values = pd.concat(self.data_values, ignore_index=True).rename(columns=cols)


     def process_trc(self, filename):
          """
          Read trc file data
          Parameters:
               filedir (str): file directory
          Returns:
               dataframe: dataframe with file raw data
          """

          seg_micromed = MicromedIO(filename)
          hem_data = seg_micromed.read_segment()
          hem_signal = hem_data.analogsignals[0]
          ch_list = seg_micromed.header['signal_channels']['name']
          find_idx = [hch for hch in range(len(ch_list)) if 'ecg' in ch_list[hch].lower()]
          self.data_values = pd.DataFrame(np.array(hem_signal[:, find_idx]), columns=[ch_list[find_idx]])
          self.columns = [ch_list[find_idx]]
          self.start_date = hem_data.rec_datetime
          self.fs = float(hem_signal.sampling_rate)


     def process_edf(self, filename):

          # read edf data
          hsm_data = read_raw_edf(filename)
          # get channels that correspond to type (POL Ecg = type ecg)
          channel_list = [hch for hch in hsm_data.ch_names if 'ecg' in hch.lower()]
          # initial datetime
          self.start_date = hsm_data.info['meas_date'].replace(tzinfo=None)
          # sampling frequency
          self.fs = hsm_data.info['sfreq']
          # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
          hsm_sig = hsm_data[channel_list]
          self.data_values = pd.DataFrame(hsm_sig[0].T, columns=channel_list)

     def process_eeg(self, filename):
          
          # read eeg data
          hsm_data = read_raw_nihon(filename)
          # get channels that correspond to type (POL Ecg = type ecg)
          channel_list = [hch for hch in hsm_data.ch_names if 'ecg' in hch.lower()]
          if len(channel_list) == 0: 
               if 'CRIH' in filename:  
                    channel_list = ['X1', 'X6'] 
               else:     
                    channel_list = ['X1', 'X6'] 
       
                    print('deal')
          
          # initial datetime
          self.start_date = hsm_data.info['meas_date'].replace(tzinfo=None)
          # sampling frequency
          self.fs = hsm_data.info['sfreq']
          # structure of hsm_sig is two arrays, the 1st has one array for each channel and the 2nd is an int-time array
          hsm_sig = hsm_data[channel_list]
          self.data_values = pd.DataFrame(hsm_sig[0].T, columns=channel_list)


if __name__ == "__main__":

    file_name = 'data/Bitalino/A2022-07-09 21-38-43.txt'
    
    LargeFileProcessor = OpenFileProcessor(mode='EPIBOX')
    time_start = time.time()
    
    LargeFileProcessor.process_chunks(file_name)
    # data_loading_utils.read_lines_from_file_as_data_chunks(file_name, chunk_size=LargeFileProcessor.CHUNK_SIZE, callback=LargeFileProcessor.process_lines)
    print('time taken: ', time.time() - time_start)

    # process_lines method is the callback method. 
    # It will be called for all the lines, with parameter data representing one single line of the file at a time