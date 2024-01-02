
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import biosppy.signals as bp
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd
import os
import pickle
from scipy import interpolate

def normal(sg, minsg, maxsg):
    res = 2 * (sg - minsg) / (maxsg - minsg) - 1
    return res


def find_extremes(sig, mode, th):
    indexes, values = bp.tools.find_extrema(sig, mode)
    ind, peaks = [], []
    for pk in range(len(values)):
        if abs(values[pk] - np.mean(sig)) > float(th):
            ind += [indexes[pk]]
            peaks += [values[pk]]
    return ind, peaks


def deep_breath(sig):
    #the signal of the first apnea is received. The deep breath is characterized by a
    #sudden increase, which corresponds to a high value of the positive derivative.
    #to find the deep breathe we use find extremes
    ind, peaks = find_extremes(sig, 'both', 0.3)
    if peaks == []:
        plt.plot(sig)
        plt.scatter(ind,peaks)
        plt.show()
    div1 = np.diff(peaks)
    deep_b = np.max(div1)
    if deep_b > 0:
        print('\ndeep breath was found')
        deep_idx = int(np.where(div1 == deep_b)[0])
        res = [peaks[deep_idx], peaks[deep_idx+1]]
    else:
        print(peaks, div1)
        res = 0
    print(res)
    return res


def autoencoder_params(encoding_dim_,input_len, list_layers, a_fun, optimizer, loss
                       ,x_train,y_train,x_test,y_test,label, epochs):
    # this is the size of our encoded representations
    encoding_dim = encoding_dim_  # 32 floats -> compression of factor 32, assuming the input is 1024 floats

    if sorted(list_layers, key=int, reverse=True) != list_layers:
        print('\nList should be in order!\n')
        popo
    input_sig = Input(shape=(input_len,))
    encoded = Dense(list_layers[0], activation = a_fun)(input_sig)
    network = list_layers + list_layers[::-1][1:] + [input_len]
    for nt in list_layers[1:]:
        encoded = Dense(nt, activation=a_fun)(encoded)
    # "decoded" is the lossy reconstruction of the input
    inverse_layers = list_layers[::-1][1:] + [input_len]
    decoded = Dense(inverse_layers[0], activation=a_fun)(encoded)
    for ll in inverse_layers[1:]:
        decoded = Dense(ll, activation=a_fun)(decoded)

    autoencoder = Model(input_sig, decoded)
    ##Let's also create a separate encoder model as well as the decoder model:
    encoder = Model(input_sig, encoded)
    encoded_input = Input(shape=(encoding_dim,))

    dec_layers = autoencoder.layers[len(list_layers)+1:]
    decoder_output = dec_layers[0](encoded_input)
    for dl in dec_layers[1:]:
        decoder_output = dl(decoder_output)
    #decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(encoded_input))))))
    decoder = Model(encoded_input,decoder_output)

    ##First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
    autoencoder.compile(optimizer=optimizer,loss=loss)

    print('\nAutoencoder Created:')
    print('Layers: ' + str(list_layers))
    print('Input Length: ' + str(input_len))
    print('Compression: ' + str(encoding_dim))
    print('Activation: ' + str(a_fun))
    print('Optimizer: ' + str(optimizer))
    print('Loss: ' + str(loss) +'\n')

    autoencoder.fit(x_train,y_train,
                    epochs=epochs,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_test, y_test))

    pickle.dump(autoencoder, open(label + '_autoencoder', 'wb'))
    pickle.dump(decoder, open(label+'_decoder', 'wb'))
    pickle.dump(encoder, open(label+'_encoder', 'wb'))
    return autoencoder, encoder, decoder


def stand(sg):
    array = (sg - np.mean(sg)) / np.std(sg)
    array_lim = [ar if abs(ar) <=1 else 1 if ar>0 else -1 for ar in array]
    return array_lim


def dist(sig):
    """ Calculates the total distance traveled by the signal,
    using the hipotenusa between 2 datapoints
    Parameters
    ----------
    s: array-like
      the input signal.
    Returns
    -------
    signal distance: if the signal was straightened distance
    """
    df_sig = abs(np.diff(sig))
    return np.sum([np.sqrt(1 + df ** 2) for df in df_sig])


def ecg_timeseries(all_files, sr, win, div, filt = True, lab = 'ecg'):

    x_train, x_test = [], []
    Y_ = []
    #ONLY ONE FILE
    print('Files are loading ...\n')
    counter = 0
    for af in [38]:#range(len(all_files)):
        markers = []

        results_list = []
        print(all_files[af])
        each_train =[]
        each_y = []
        #df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        #import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0,sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.4], sampling_rate=sr)['signal']
        ecg_ = bp.ecg.ecg(df.A2, show=False)
        ecg_rate = ecg_['heart_rate']
        ecg_sig = ecg_['filtered']
        idx_peaks = ecg_['rpeaks']
        print(idx_peaks)
        ecg_peaks = [ecg_sig[e]  for e in range(len(ecg_sig)) if e in idx_peaks]
        interp = interpolate.interp1d(idx_peaks, ecg_peaks, kind='quadratic')
        ts_rate = ecg_['heart_rate_ts']
        bvp_ = bp.bvp.bvp(df.A4, show=False)
        eda_ = bp.eda.eda(df.A3,show=False)['filtered']
        bvp_rate= bvp_['heart_rate']
        bvp_ts= bvp_['heart_rate_ts']
        #X = normal(np.array(X))
        # the vector is segmented into samples of fixed size 2000
        # the samples are resized to a length of 1024 to fit in the autoencoder
        X_cut = [signal.resample(X[i:i + win],1024) for i in range(0, len(X)-win, win)]
        #X_cut = [X[i:i + win] for i in range(0, len(X)-win, win)]
        # the first 20% goes to test and the rest goes to training
        list_markers = [[mk,marker] for mk, marker in enumerate(df.MARKER) if marker != 0 and marker in marker_label.keys()]
        #print(list_markers)
        #print(df.columns)
        deep = X[list_markers[4][0]:list_markers[5][0]]
        xminmax = deep_breath(deep)

    #plt.savefig('ALL', bbox_inches='tight', format='png')


        for j in range(0,len(list_markers),2):

            sig = normal(X[list_markers[j][0]:list_markers[j+1][0]],xminmax[0],xminmax[1])
            unsig = X[list_markers[j][0]:list_markers[j+1][0]]
            #sig = normal(unsig, np.min(unsig), np.max(unsig))
            markers += [list_markers[j][0]/1000]
            markers += [list_markers[j+1][0]/1000]
            indexes, peaks = find_extremes(sig, 'both', 0.1)
            #plt.plot(sig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig),4)))
            #plt.plot(unsig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig),4)))
            #plt.scatter(indexes,peaks)
            #print(len(sig))
            win = 10000
            results_list += [0]
            for k in range(0, len(sig)-win,win):
                #print('='*50, list_markers[j][1], k)
                #print(j)
                _peaks = [peaks[idx] for idx in range(len(indexes)) if k <= indexes[idx] <= k+win]
                _indexes = [idx for idx in indexes if k<=idx<=k+win]

                if _indexes:
                    result = dist(_peaks)*len(_peaks)*2500/(_indexes[-1]-_indexes[0])
                    #plt.scatter(_indexes,_peaks)
                else:
                    result = 'No peaks'
                if 'Ap' in marker_label[list_markers[j][1]]:
                    if str(result) == 'nan' or result == 'No peaks':
                        each_train += [signal.resample(normal(unsig[k  :k+win],xminmax[0],xminmax[1]),1000)]
                        each_y += ['Apneia']
                        counter += 1
                        #print(result,each_y[-1])
#                     plt.subplot(2,1,1)
#                         plt.ylim((-1,1))
#                         plt.plot(range(1000), each_train[-1], label=(marker_label[list_markers[j][1]], result))

#                     plt.legend()
                       # plt.show()
                        results_list += [20]*10
                    else:
                        results_list += [-100]*10
                        #print(result,'Removed')
                else:
                    if str(result) != 'nan' or result != 'No peaks':
                        results_list += [20]*10
                        #print(result,'Removed')
                    #else:
                        #each_train += [signal.resample(normal(unsig[k:k+win],xminmax[0],xminmax[1]),1000)]
                        each_train += [signal.resample(normal(int_sig))]
                        if lab=='layer1':
                            each_y += ['Breathe']
                        if lab == 'layer2':
                            each_y += [marker_label[list_markers[j][1]]]
                        #print(result,each_y[-1])
#                     plt.subplot(2,1,1)
#                     plt.plot(range(1000),each_train[-1], label=(marker_label[list_markers[j][1]], result))
#                     #plt.ylim((-1,1))
#                     plt.legend()
#                     plt.show()
                    else:
                        results_list += [-100]*10

        x_train += [each_train]
        Y_ += [each_y]

        plt.figure(figsize=(30,10))
        #plt.subplot(3, 1, 1)

        plt.title('Heart Rate')
        #plt.plot(ts_rate, normal(ecg_rate, np.min(ecg_rate),np.max(ecg_rate)))
        int_sig = interp(np.arange(idx_peaks[0],idx_peaks[-1]))
        plt.title('R Peaks Interpolation')
        plt.ylim(-2,2)
        plt.plot(ts_rate, signal.resample(normal(ecg_rate, np.min(ecg_rate), np.max(ecg_rate)),len(ecg_rate)), label='HRV')
        plt.plot(ts_rate, signal.resample(normal(int_sig, np.min(int_sig), np.max(int_sig)), len(ecg_rate)), label ='R Interpolation')
        plt.plot(ts_rate, signal.resample(normal(X, np.min(X), np.max(X)), len(ecg_rate)),label='Respiration')

        plt.vlines(markers, -1,1)
        plt.legend()
        plt.savefig('AL' + str(af) + '.png', bbox_inches='tight', format='png')

        plt.subplot(3,1,2)
        #plt.plot(bvp_ts, normal(bvp_rate, np.min(bvp_rate),np.max(bvp_rate)))

        #plt.plot(ts_rate, signal.resample(normal(int_sig,np.min(int_sig),np.max(int_sig)),len(ecg_rate)))
        #plt.plot(ts_rate,signal.resample(normal(bvp_rate,np.min(bvp_rate),np.max(bvp_rate)),len(ecg_rate)))
        #plt.plot(ts_rate, signal.resample(normal(eda_, np.min(eda_), np.max(eda_)), len(ecg_rate)))
        plt.vlines(markers, -1, 1)
        plt.subplot(3,1,3)
        plt.ylim(-1,1)
        plt.title('Respiration Signal')
        plt.plot(ts_rate, signal.resample(normal(X,np.min(X),np.max(X)),len(ecg_rate)))

        #plt.plot(ts_rate, signal.resample(normal(int_sig, np.min(int_sig), np.max(int_sig)), len(ecg_rate)))
        plt.xlabel('time (s)')
        plt.vlines(markers, -1, 1)
        plt.ylabel('Amplitude')
        #plt.show()

        #print(each_y)
    print(counter)

    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))

    return np.array(x_train), np.array(Y_)


def find_nearest(array, value):

    return np.argmin(np.abs(array-value/1000))


def load_timeseries2(all_files, sr, win, div, lab):

    x_train = []
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        each_y = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.35], sampling_rate=sr)[
            'signal']
        #ppg = bp.tools.filter_signal(df.A4, ftype='butter', band='bandpass',
        #                           order=4, frequency=[1, 8], sampling_rate=sr)['signal']
        # X = normal(np.array(X))

        # the vector is segmented into samples of fixed size 2000
        # the samples are resized to a length of 1024 to fit in the autoencoder
        X_cut = [signal.resample(X[i:i + win], 1024) for i in range(0, len(X) - win, win)]
        # X_cut = [X[i:i + win] for i in range(0, len(X)-win, win)]
        # the first 20% goes to test and the rest goes to training
        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]
        # print(list_markers)
        # print(df.columns)
        #resp = bp.resp.ecg_derived_respiration(df.A2, X, show=True)
        #ecg_ = bp.ecg.ecg(df.A2, show=False)
        #ecg_rate = normal(ecg_['heart_rate'],np.min(ecg_['heart_rate']),np.max(ecg_['heart_rate']))
        #ecg_sig = ecg_['heart_rate']
        #eee_filt = ecg_['filtered']
        #ts_rate = ecg_['heart_rate_ts']
        #idx_peaks = ecg_['rpeaks']
        #print(idx_peaks)
        #print(eee_filt)
        #ecg_peaks = [eee_filt[e]  for e in range(len(eee_filt)-1) if e in idx_peaks]
        #print(ecg_peaks)
        #interp = interpolate.interp1d(idx_peaks, ecg_peaks, kind='quadratic')
        #bvp_ = bp.bvp.bvp(df.A4, show=False)
        #bvp_hr = bvp_['heart_rate']
        #bvp_hrts = bvp_['heart_rate_ts']
        deep = X[list_markers[4][0]:list_markers[5][0]]
        xminmax = deep_breath(deep)
        #ecg_ext = deep_breath(ecg_rate) #[find_nearest(ts_rate, list_markers[4][0])-5000:find_nearest(ts_rate, list_markers[5][0])])
        #norm_ecg = normal(bvp_hr, np.min(bvp_hr), np.max(bvp_hr))
        #eda_sig = bp.eda.eda(df.A3, show=False)['filtered']
        #norm_eda = normal(eda_sig, np.min(eda_sig), np.max(eda_sig))

        #int_sig = interp(np.arange(idx_peaks[0], idx_peaks[-1]))

        for j in range(0, len(list_markers), 2):

            sig = normal(X[list_markers[j][0]:list_markers[j + 1][0]], xminmax[0], xminmax[1])
            unsig = X[list_markers[j][0]:list_markers[j + 1][0]]
            #eda_ = norm_eda[list_markers[j][0]:list_markers[j + 1][0]]

            # sig = normal(unsig, np.min(unsig), np.max(unsig))
            indexes, peaks = find_extremes(sig, 'both', 0.2)
            #plt.plot(sig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig), 4)))
            for k in range(0, len(sig) - win, int(win/2)):
                # print(j)
                #ecg = norm_ecg[
                 #     find_nearest(ts_rate, (list_markers[j][0]+k)*0.001):find_nearest(ts_rate, 0.001*(list_markers[j][0]+k+win))]
                #intp = int_sig[
                 #     find_nearest(ts_rate, list_markers[j][0]+k):find_nearest(ts_rate, list_markers[j][0]+k+win)]

                _peaks = [peaks[idx] for idx in range(len(indexes)) if k <= indexes[idx] <= k + win]
                _indexes = [idx for idx in indexes if k <= idx <= k + win]

                if _indexes:
                    result = dist(_peaks) * len(_peaks) * 2500 / (_indexes[-1] - _indexes[0])
                    # plt.scatter(_indexes,_peaks)
                else:
                    result = 'No peaks'
                print(result)
                #"""

                if 'Ap' in marker_label[list_markers[j][1]]:
                    if str(result) == 'nan' or result == 'No peaks':
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        #each_train += [signal.resample(eda_[k:k+win],1000)]
                        #each_train += [signal.resample(intp, 1000)]
                        each_y += ['Apneia']
                        counter += 1

                    else:
                        rej += 1
                    # print(result,'Removed')
                    #"""
                #if 'Relax' in marker_label[list_markers[j][1]]:

                else:

                    if str(result) != 'nan' or result != 'No peaks':
                        # print(result,'Removed')
                        # else:

                        #each_train += [signal.resample(normal(intp, np.min(int_sig), np.max(int_sig)),1000)]
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        #each_train += [signal.resample(eda_[k:k+win], 1000)]
                        each_y += ['Breathe']
                        #counter+= 1
                        #each_y += [marker_label[list_markers[j][1]]]
                        # print(result,each_y[-1])
                    #else:
                     #   rej += 1

        x_train += [each_train]

        Y_ += [each_y]
        # print(each_y)
    print(counter, rej)

    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))


def load_timeseries3(all_files, sr, win, div, lab):

    x_train = []
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        each_y = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.35], sampling_rate=sr)[
            'signal']
        #ppg = bp.tools.filter_signal(df.A4, ftype='butter', band='bandpass',
        #                           order=4, frequency=[1, 8], sampling_rate=sr)['signal']
        # X = normal(np.array(X))

        # the vector is segmented into samples of fixed size 2000
        # the samples are resized to a length of 1024 to fit in the autoencoder
        X_cut = [signal.resample(X[i:i + win], 1024) for i in range(0, len(X) - win, win)]
        # X_cut = [X[i:i + win] for i in range(0, len(X)-win, win)]
        # the first 20% goes to test and the rest goes to training
        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]

        #deep = X[list_markers[0][0]:list_markers[1][0]]
        #xminmax = deep_breath(deep)
        xminmax = [np.min(X),np.max(X)]


        for j in range(0, len(list_markers), 2):
            unsig = X[list_markers[j][0]:list_markers[j + 1][0]]

            for k in range(0, len(unsig) - win, int(win/2)):
                if 'Ap' in marker_label[list_markers[j][1]]:
                    each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                    each_y += ['Apneia']
                    counter += 1

                else:
                    each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                    each_y += ['Breathe']

        x_train += [each_train]

        Y_ += [each_y]
    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))


def real_timeseries(all_files, sr, win, div, lab):
    x_train, x_test = [], []
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter = 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        each_y = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = \
        bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.4], sampling_rate=sr)[
            'signal']

        list_markers = [[mk, marker] for mk, marker in enumerate(df.MARKER) if
                        marker != 0 and marker in marker_label.keys()]

        deep = X[list_markers[4][0]:list_markers[5][0]]
        xminmax = deep_breath(deep)

        for j in range(0, len(list_markers), 2):

            sig = normal(X[list_markers[j][0]:list_markers[j + 1][0]], xminmax[0], xminmax[1])
            if j == 0:
                outsig = X[0:list_markers[j][0]]
            else:
                outsig = X[list_markers[j - 1][0]:list_markers[j][0]]
                print('I pass here')
            unsig = X[list_markers[j][0]:list_markers[j + 1][0]]
            # sig = normal(unsig, np.min(unsig), np.max(unsig))
            indexes, peaks = find_extremes(sig, 'both', 0.1)
            plt.plot(sig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig), 4)))
            # plt.plot(unsig, label=(marker_label[list_markers[j][1]], np.round(np.var(sig),4)))
            # plt.scatter(indexes,peaks)
            # print(len(sig))
            win = 10000
            for ou in range(0, len(outsig) - win, int(win / 2)):
                each_train += [signal.resample(normal(outsig[ou:ou + win], xminmax[0], xminmax[1]), 1000)]
                each_y += ['Removed']
            for k in range(0, len(sig) - win, int(win / 2)):

                print('=' * 50, list_markers[j][1], k)
                # print(j)
                _peaks = [peaks[idx] for idx in range(len(indexes)) if k <= indexes[idx] <= k + win]
                _indexes = [idx for idx in indexes if k <= idx <= k + win]

                if _indexes:
                    result = dist(_peaks) * len(_peaks) * 2500 / (_indexes[-1] - _indexes[0])
                    # plt.scatter(_indexes,_peaks)
                else:
                    result = 'No peaks'

                if 'Ap' in marker_label[list_markers[j][1]]:
                    if str(result) == 'nan' or result == 'No peaks':
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_y += ['Apneia']
                        counter += 1
                        # print(result,each_y[-1])
                    #                     plt.subplot(2,1,1)
                    #                     #plt.ylim((-1,1))
                    #                     plt.plot(range(1000), each_train[-1], label=(marker_label[list_markers[j][1]], result))

                    #                     plt.legend()
                    #                     plt.show()
                    else:
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_y += ['Removed']
                        # print(result,'Removed')
                else:
                    if str(result) != 'nan' or result != 'No peaks':
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_y += ['Removed']
                        # print(result,'Removed')
                    else:
                        each_train += [signal.resample(normal(unsig[k:k + win], xminmax[0], xminmax[1]), 1000)]
                        each_y += ['Breathe']
                        #each_y += [marker_label[list_markers[j][1]]]
                        # print(result,each_y[-1])


        x_train += [each_train]
        Y_ += [each_y]
        # print(each_y)
    print(counter)

    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))
    pickle.dump(np.array(Y_), open(lab + '_Y', 'wb'))


def load_bitalino(all_files, sr, win, div, lab):
    x_train = np.array()
    Y_ = []
    # ONLY ONE FILE
    print('Files are loading ...\n')
    counter, rej = 0, 0
    for af in range(len(all_files)):
        print(all_files[af])
        each_train = []
        # df = pd.DataFrame.read_csv(all_files[af], sep=';', names=['DateTime', 'X'])
        # import pandas
        df = pd.DataFrame.from_csv(all_files[af], parse_dates=True, index_col=0, header=0, sep=';')
        # the vector is normalized
        X = bp.tools.filter_signal(df.A1, ftype='butter', band='bandpass', order=2, frequency=[0.1, 0.35], sampling_rate=sr)[
            'signal']

        deep = X[:]
        xminmax = deep_breath(deep)

        for k in range(0, len(X) - win, int(win)):
            each_train += [signal.resample(normal(X[k:k + win], xminmax[0], xminmax[1]), 1000)]

        x_train.append(np.array(each_train))
    print('Finished')
    pickle.dump(np.array(x_train), open(lab + '_X', 'wb'))


if __name__ == '__main__':

    folder = "C:\\Users\\Mariana\\Documents\\Databases\\APNEIA\\"
    # all_files = [folder+ol for ol in os.listdir(folder)]
    all_files = [folder + ol for ol in os.listdir(folder)]
    # samplint rate, window size of each sample
    sampling_rate = 1000.
    window = 10000
    # percentage of the dataset that will be used for testing
    div = 0.2


    marker_label = dict([(2, 'Relax'), (5, 'Sinus'), (8, '1Ap'), (10, '2Ap'), (12, '3Ap'), (14, '4Ap'), (16, '5Ap')])


    #ecg_timeseries(all_files, sampling_rate, window, div, lab='intp')
    #load_timeseries3(all_files, sampling_rate, window, div, lab='all')
    #load_timeseries2(all_files,sampling_rate,20000,div,lab='20')
    load_bitalino(all_files, sampling_rate, window, div, lab='bit')
    # x_apneia_train, y_apneia_train = load_timeseries(all_files, sampling_rate, window, div, filt = False, lab = 'layer2')

    #load_timeseries(all_files, sampling_rate, window, 'MaxMin', 'Binary10')

