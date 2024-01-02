# Autoencoder Model for Respiratory Signals
# This file contains the functions to train and validate the autoencoder model
# created by: Mariana Abreu
# date: 15 December 2023

import pickle

from keras.layers import Input, Dense
from keras.models import Model


def autoencoder_params(encoding_dim_,input_len, list_layers, a_fun, optimizer, loss
                       ,x_train, y_train, x_test, y_test, label, epochs):
    """This function will train the autoencoder

    Args:
        encoding_dim_ (int): size of shortest layer (bottleneck)
        input_len (int): size of input (datapoints)
        list_layers (list): number of nodes in each layer by order of layers of the encoding part
        a_fun (str): activation function, in this case usually tanh 
        optimizer (str): optimizer, usually adam
        loss (str): loss function, it depends on the input 
        x_train (array): data to train the autoencoder
        y_train (array): usually is equal to x_train
        x_test (array): data to validate the autoencoder
        y_test (array): usually is equal to x_test
        label (str): label to save the encoder and decoder
        epochs (int): number of epochs to run 
    """
    # this is the size of our encoded representations
    encoding_dim = encoding_dim_  # 32 floats -> compression of factor 32, assuming the input is 1024 floats
    assert(sorted(list_layers, key=int, reverse=True) == list_layers)

    input_sig = Input(shape=(input_len,))
    encoded = Dense(list_layers[0], activation = a_fun)(input_sig)

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

    pickle.dump(decoder, open(label+'_decoder', 'wb'))
    pickle.dump(encoder, open(label+'_encoder', 'wb'))

    return decoder, encoder


def create_autoencoder(x_train, y_train, x_test, y_test, label, loss,
                       activ='tanh', opt='adam', nodes=[500, 250, 50], epochs=100):
    """Training and validation of the autoencoder model. This function has some useful default parameters

    Args:
        x_train (array): data to be trained
        y_train (array): output data to be trained
        x_test (array): validation input data
        y_test (array): validation output data
        label (str): label to save the trained model parameters
        loss (str): loss function, it depends on the input 
        activ (str, optional): activation function. Defaults to 'tanh'.
        opt (str, optional): optimizer. Defaults to 'adam'.
        nodes (list, optional): number of nodes in each layer by order of layers of the encoding part. Defaults to [500, 250, 50].
        epochs (int, optional): number of epochs. Defaults to 100.

    Returns:
        models: decoder and encoder models trained.
    """
    decoder, encoder = autoencoder_params(nodes[-1], x_train.shape[1], nodes, activ, opt, loss,
                          x_train, y_train, x_test, y_test, label, epochs)
    return decoder, encoder