from os import listdir

import numpy as np
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPool2D
from keras.layers import Dense, Input, Conv2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import LSTM, Embedding, Dense, Dropout, Activation, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import *


def one_hot(x_):
    index = 0
    temp = np.zeros([x_.shape[0], 12])
    for item in x_:
        temp[index][int(item) - 1] = 1
        index += 1
    return temp
InputImg = []
files = listdir('img')
for file in range(1, 8985):
    InputImg.append(img_to_array(load_img('img/' + '{}.jpg'.format(file))))
InputImg = np.reshape(InputImg, [-1, 3, 36, 36]) / 255

input_img = Input(shape=(3, 36, 36,))
conv1 = Conv2D(32, (3, 3), input_shape=(3, 36, 36), data_format='channels_first', activation='relu')(input_img)
max_pool1 = MaxPool2D(data_format='channels_first')(conv1)

conv2 = Conv2D(32, (3, 3), data_format='channels_first', activation='relu')(max_pool1)
max_pool2 = MaxPool2D(data_format='channels_first')(conv2)

conv3 = Conv2D(32, (3, 3), data_format='channels_first', activation='relu')(max_pool2)
reshape = Reshape((-1, 800))(conv3)


fc1 = Dense(128, activation='relu')(reshape)
fc2 = Dense(128, activation='relu')(fc1)
CNN = Model(inputs=input_img, outputs=fc2)
CNN.compile(optimizer='adadelta', loss='binary_crossentropy')
# CNN.fit(InputImg, InputImg, nb_epoch=25, batch_size=32, shuffle=True)
vec_ = CNN.predict(InputImg)
vec_ = np.squeeze(vec_)
# print(vec_)
test = np.load('zh_simplified_test.npy')  # 118615
train = np.load('zh_simplified_train.npy')  # 355843
# print(test[1])
# print(train.shape)
x_train = []
y_train = []
x_test = []
y_test = []
MAX_LEN = 71
for x in test:
    x = x.split('\t')
    x_test.append(x[1].split(','))
    y_test.append(x[0])
x_test = np.array(x_test)
y_test = np.array(y_test)
for x in train:
    x = x.split('\t')
    x_train.append(x[1].split(','))
    y_train.append(x[0])
x_train = np.array(x_train)
y_train = np.array(y_train)
del test, train
y_train = one_hot(y_train)
y_test = one_hot(y_test)
x_train_padded = pad_sequences(x_train, maxlen=71)
x_test_padded = pad_sequences(x_test, maxlen=71)
# x_train_padded = np.expand_dims(x_train_padded, -1)
# x_test_padded = np.expand_dims(x_test_padded)
text_input = Input(shape=(71,))
vec__ = np.arange(1, 8986)
vec__ = np.expand_dims(vec__, -1)

embedding_layer = Embedding(
        8985,
        1,
        weights=[vec__],
        input_length=71,
        trainable=False
    )(text_input)
GRU_layer = GRU(64)(embedding_layer)
dense_ = Dense(12, input_shape=(None, 64))(GRU_layer)
dropout = Dropout(0.5)(dense_)
relu_layer = Activation('relu')(dropout)


GRU_model = Model(inputs=text_input, outputs=relu_layer)
GRU_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall_threshold(0)])
# print(GRU_model.layers)
history_result = GRU_model.fit(x_train_padded, y_train, batch_size=1024, epochs=5, validation_split=0.33)
score = GRU_model.evaluate(x_test_padded, y_test, batch_size=1024)
np.save('CNN_embedding.npy', vec_)
with open('result/CNN_embedding.txt', 'w') as f:
    f.writelines(str(score))
print('test_score:{} test_accuracy:{} test_recall:{}'.format(score[0], score[1], score[2]))
with open('CNN_score.txt', 'w') as f:
    f.write('test_score:{} test_accuracy:{} test_recall:{}'.format(score[0], score[1], score[2]))

