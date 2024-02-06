# ResNet model for time series classification
#
# Reference: João Saraiva Thesis
# Created by: Mariana Abreu
# Created on: 22/11/2023


from keras.models import Input, Model
from keras.layers import Conv1D, Add, Activation, Flatten, Dense, Dropout, GlobalAveragePooling1D, ReLU, MaxPooling1D, Add
from keras.layers.normalization import BatchNormalization
from keras.activations import sigmoid
from keras.optimizers import Adam


kernel_size= 16
dropout_rate = 0.5

def resnet_algorithm(x, R, D, F, F0, kernel_size):
    """
    ResNet algorithm
    :param x: input
    :param R: number of residual blocks
    :param D: dropout rate
    :param F: filter factor
    :param 
    :param F0: initial number of filters
    :param kernel_size: kernel size
    :return: output
    """
    filters = F0
    # Input Block
    x = Conv1D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Residual Blocks
    for r in range(R):
        identity = x
        if r != 0:
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Dropout(D)(x)
        # Upfilter
        if r % F == 1:
            filters *= 2
        # Downsample
        if r % D == 1:
            x = Conv1D(filters, kernel_size, strides=2, padding='same')(x)
            identity = MaxPooling1D(pool_size=2)(identity)
        else:
            x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(D)(x)
        x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
        # Shortcut
        x = Add()([x, identity])
    # Classification Block
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(1)(x)
    x = sigmoid(x)
    return x

R = 17
D = 6
F = 6
F0 = 64
N = 35 #seconds
No = 0.5
PH = 30 #minutes
L0 = 0.01
B = 32
# Define your input shape
input_shape = (None, None)  # replace with your input shape
inputs = Input(shape=input_shape)

# Call your ResNet function
outputs = resnet_algorithm(inputs, R=R, D=D, F=F, F0=F0, kernel_size=16)

# Create your model
model = Model(inputs=inputs, outputs=outputs)

# Compile your model
model.compile(optimizer=Adam(learning_rate=L0), loss='binary_crossentropy')
