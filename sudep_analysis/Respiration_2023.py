# Train autoencoders on respiration data from the RGBT training dataset
# Apply the trained autoencoders to the data from the PreEpiSeizures dataset
# Evaluate output/input correlation and compare with ECG quality along the same segments

# import packages

# built-in libraries
import os
import pickle

# third-party libraries
import biosppy as bp
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import pearsonr

# local libraries
import utils_classification as uc


def load_rgbt_data(dir='', again=True, overlap=0.95):
    """
    Load the RGBT data from the training dataset
    This data is saved in users_data pickle
    The labels are saved in user_labels pickle
    :param dir: directory where the data is stored
    :param again: whether to load the data again
    :return: users_data, user_labels
    """
    if again:
        # load the raw data again
        # marker labels to help identify the respiration exercises
        marker_label = dict([(2, 'N'), (5, 'N'), (8, 'A'), (10, 'A'), (12, 'A'), (14, 'A'), (16, 'A')])
        _label = dict([(0, 'N'), (1, 'N'), (2, 'A'), (3, 'A'), (4, 'A'), (5, 'A'), (6, 'A')])
        # directory where the raw data is stored
        bitalino_dir = '/Users/saraiva/dev/Respiration/APNEIA'
        # list of files in the directory
        bitalino_files = [os.path.join(bitalino_dir, f) for f in os.listdir(bitalino_dir)]
        # iterate over the files
        respiration_data = pd.DataFrame()
        for bf in range(len(bitalino_files)):
            # get data from one acquisition (one patient)
            df = pd.read_csv(bitalino_files[bf], parse_dates=True, index_col=0, header=0,sep=';')
            # filter the respiratory signal
            resp = np.array(bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.01,0.35],sampling_rate=1000.)[
                                'signal'])
            user_data = pd.DataFrame({'RESP': resp})
            #user_data['normalized_RESP_100'] = 100 * (user_data.RESP - np.min(user_data.RESP)) / (
            #        np.max(user_data.RESP) - np.min(user_data.RESP))
            # get the segments for each exercise
            list_markers = np.array([[mk, marker] for mk, marker in enumerate(df.MARKER) if
                                marker != 0 and marker in marker_label.keys()])
            # free breathing
            free_breathing_index = list_markers[list_markers[:, 1] == 2, 0]
            free_data = user_data.iloc[free_breathing_index[0]:free_breathing_index[1]]['RESP']
            free_data_segments = [free_data.iloc[i: i + 1000*60] for i in range(0, len(free_data)-int(1000*60), int(1000*60*(1-overlap)))]
            # sinus breathing
            sinus_breathing_index = list_markers[list_markers[:, 1] == 5, 0]
            sinus_data = user_data.iloc[sinus_breathing_index[0]:sinus_breathing_index[1]]['RESP']
            sinus_data_segments = [sinus_data.iloc[i: i + 1000*60] for i in range(0, len(sinus_data)-int(1000*60), int(1000*60*(1-overlap)))]
            # apnea
            apnea_data_segments = []
            for index in [8, 10, 12, 14, 16]:

                apnea_index = list_markers[list_markers[:, 1] == index, 0]
                apnea_data_segments += [user_data.iloc[apnea_index[0]:apnea_index[0] + 1000*60]['RESP']]
            
            # only accept segments with 60 seconds
            free_data_segments = list(filter(lambda x: len(x) == 1000*60, free_data_segments))
            sinus_data_segments = list(filter(lambda x: len(x) == 1000*60, sinus_data_segments))
            respiration_segments = np.vstack((free_data_segments, sinus_data_segments))
            # resample to 1000 points
            respiration_segments = resample(respiration_segments, 1000, axis=1)
            respiration_df = pd.DataFrame(respiration_segments)
            respiration_df_norm = respiration_df.T.apply(lambda x: 100*(x - x.min())/(x.max()-x.min()))
            respiration_df = pd.DataFrame(respiration_df_norm.T)
            respiration_df['Label'] = np.hstack((['Free'] * (len(free_data_segments)), ['Sinus'] * (len(sinus_data_segments))))
            respiration_df['Segment'] = np.hstack((np.arange(len(free_data_segments)), np.arange(len(sinus_data_segments))))
            respiration_df['User'] = bf
            print('File: ', bitalino_files[bf])
            respiration_data = pd.concat((respiration_data, respiration_df), ignore_index=True)
        # save the data
        respiration_data.rename(columns={i:str(i) for i in range(respiration_segments.shape[1])}, inplace=True)
        respiration_data.to_parquet(os.path.join('data', 'respiration_data.parquet'), engine='fastparquet')
    else:
        # load the data
        #users_data = pickle.load(open(os.path.join(dir, 'user_data'), 'rb'))
        # load labels
        #user_labels = pickle.load(open(os.path.join(dir, 'user_labs'), 'rb'))
        # transform the list of users data into a numpy array
        #users_data = np.array(users_data)
        #user_labels = np.array(user_labels)
        #raise IOError('Not implemented yet - Transformation to dataframe')
        respiration_data = pd.read_parquet(os.path.join('data', 'respiration_data.parquet'))

    return respiration_data


def respiration_training(rgbt_data, test_size=0.2, again = False):
    """
    Train the AutoEncder with Respiration Segments
    :param rgbt_data: RGBT data
    :return: respiration_data
    """
    label = 'RGBT'
    if again:
        # get the indices of the respiration segments
        loss = 'cosine_similarity'
        activ ='tanh'
        opt ='adam'
        
        epochs = 300
        nodes = [500, 250, 50]
        # get the test users
        test_users = rgbt_data['User'].unique()[-int(len(rgbt_data['User'].unique()) * test_size):]
        # get the training and test data
        x_test = rgbt_data.loc[rgbt_data['User'].isin(test_users)].drop(['User', 'Label', 'Segment'], axis=1)
        x_train = rgbt_data.loc[rgbt_data['User'].isin(test_users) == False].drop(['User', 'Label', 'Segment'], axis=1)
        # train the autoencoder
        modelAE, encAE, decAE = uc.autoencoder_params(nodes[-1], 1000, nodes, activ, opt, loss,
                            x_train, x_train, x_test, x_test, label, epochs)
    else: 
        # load the autoencoder
        modelAE = pickle.load(open(label + '_autoencoder', 'rb'))
        encAE = pickle.load(open(label + '_encoder', 'rb'))
        decAE = pickle.load(open(label + '_decoder', 'rb'))
    
    return modelAE, encAE, decAE

def process_segment_resp(segment):
    """
    Process one respiration segment
    :param segment: respiration segment
    :param modelAE: autoencoder model
    :return: processed segment
    """
    sampling_rate = 80

    if len(segment) != 60 * sampling_rate:
        # segment has less than 60 seconds
        return pd.DataFrame()
    # resample segment to 1000 points
    resp_data = resample(segment['RESP'], 1000)
    # normalize resp_data ? 
    norm_resp_data = 100*(resp_data-np.min(resp_data))/ (np.max(resp_data)-np.min(resp_data))
    # get the output of the autoencoder
    return pd.DataFrame([norm_resp_data], index=[segment['datetime'].iloc[0]])


def epilepsy_respiration_data(patient='BLIW', modelAE = None, encoder=None, decoder=None, timestamps=200):
    """
    Load the respiration data from the PreEpiSeizures dataset
    :param patient: patient to load
    :return: data
    """
    # get the data
    data = pd.read_parquet('data/respiration/{patient}_all_respiration_data.parquet'.format(patient=patient))
    # segment into 60s segments resampled to 1000 points
    timestamps_segments = pd.date_range(start=data['datetime'].iloc[0], end=data['datetime'].iloc[-1]-pd.Timedelta(seconds=60), freq='60s')
    data_segments = list(map(lambda x: process_segment_resp(data.loc[data['datetime'].between(x, x+pd.Timedelta(seconds=60))]), timestamps_segments))
    data_segments = pd.concat(list(filter(lambda x: not x.empty, data_segments)))
    # run the autoencoder
    output = modelAE.predict(data_segments)
    # compute the correlation between the input and output
    corr_ = np.array(list(map(lambda i: correlation(output[i], data_segments.iloc[i]), range(len(output)))))
    # dataframe with correlation points and timestamp that marks the begining of the segment
    corr_df = pd.DataFrame(corr_, index= data_segments.index, columns=[str(i) for i in range(len(corr_[0]))])
    return corr_df


def correlation(sig1, sig2, points=10):
    """
    Receive two signals of 60s with 1000 points.
    Compute the correlation vector between the two signals
    The correlation used will be the pearson correlation
    The number of correlation points is also an input
    """
    # divide the signal into k windows (k = points)
    sig1_windows = np.array(np.array_split(sig1, points))
    sig2_windows = np.array(np.array_split(sig2, points))
    # compute the correlation between the windows
    return np.array(list(map(lambda i: pearsonr(sig1_windows[i], sig2_windows[i])[0], range(points))))


if __name__ == '__main__':
    # load RGBT data
    rgbt_data = load_rgbt_data(again=False, overlap=0.8)
    print(len(rgbt_data), ' segments loaded')
    # extract respiration data
    modelAE, encAE, decAE = respiration_training(rgbt_data, again=False)
    patient_resp = [pat.split('_')[0] for pat in os.listdir('data/respiration/') if '_all_respiration' in pat]

    for patient in patient_resp:
        if patient in ['LCGM', 'RLJW', 'DQXF', 'FFRZ', 'UWEF', 'OXDN']:
            continue
        corr_points_path = 'data/respiration/{patient}_respiration_corr_points_all.parquet'.format(patient=patient)
        if os.path.isfile(corr_points_path):
            print('Correlation points already computed for patient: ', patient)
            continue
        else:
            print('Processing patient: ', patient)
            resp_ = epilepsy_respiration_data(patient=patient, modelAE=modelAE, encoder=encAE, decoder=decAE)
            resp_.to_parquet(corr_points_path, engine='fastparquet')
            print('Done')
    # train autoencoders
    # apply autoencoders to PreEpiSeizures data
    # evaluate output/input correlation
    # compare with ECG quality along the same segments