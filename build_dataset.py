from os import listdir

import numpy
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPool2D
from keras.layers import Dense, Input, Conv2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array


def build_embedding_dims_autoencoder(embedding_dims):
    InputImg = []
    files = listdir('img')
    for file in range(1, 8986):
        InputImg.append(img_to_array(load_img('img/' + '{}.jpg'.format(file))))
    InputImg = numpy.reshape(InputImg, [-1, 3 * 36 * 36]) / 255

    input_img = Input(shape=(3 * 36 * 36,))
    encoded = Dense(embedding_dims, activation='relu')(input_img)
    decoded = Dense(3888, activation='relu')(encoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(input=input_img, output=encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(InputImg, InputImg, nb_epoch=50, batch_size=32, shuffle=True)
    print('saving weights and config')
    with open('net_config.json', 'w') as f:
        f.write(autoencoder.to_json())
    numpy.save('weights.npy', autoencoder.get_weights())

def build_embedding_dims_CNN():
    InputImg = []
    files = listdir('img')
    for file in range(1, 8985):
        InputImg.append(img_to_array(load_img('img/' + '{}.jpg'.format(file))))
    InputImg = numpy.reshape(InputImg, [-1, 3, 36, 36]) / 255

    input_img = Input(shape=(3, 36, 36,))
    conv1 = Conv2D(32, (3, 3), input_shape=(3, 36, 36), data_format='channels_first', activation='relu')(input_img)
    max_pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), input_shape=(3, 36, 36), data_format='channels_first', activation='relu')(max_pool1)
    max_pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), input_shape=(3, 36, 36), data_format='channels_first', activation='relu')(max_pool2)
    reshape = Reshape((32*5*5))(conv3)

    fc1 = Dense(128, activation='relu')(reshape)
    fc2 = Dense(128, activation='relu')(fc1)
    CNN = Model(inputs=input_img, outputs=fc2)
    CNN.compile(optimizer='adadelta', loss='binary_crossentropy')
    CNN.fit(InputImg, InputImg, nb_epoch=25, batch_size=32, shuffle=True)
    print('saving weights and config')
    with open('net_config.json', 'w') as f:
        f.write(CNN.to_json())
    numpy.save('CNN_weights.npy', CNN.get_weights())
