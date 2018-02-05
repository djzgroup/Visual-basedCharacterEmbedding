from keras.layers import LSTM, Embedding, Dense, Dropout, Activation, GRU, MaxPooling1D, Conv1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.metrics import *
from keras.optimizers import *
import numpy as np
import timeit
from keras.utils.vis_utils import plot_model

das = timeit.Timer


def train_evaluate(embedding_dims, layer_id, training_size):
    def one_hot(x_):
        index = 0
        temp = np.zeros([x_.shape[0], 12])
        for item in x_:
            temp[index][int(item) - 1] = 1
            index += 1
        print(temp[0])
        return temp

    def two_layer_LSTM(dims):
        model.add(LSTM(128, input_shape=(None, dims), return_sequences=True))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32))

    def one_layer_GRU(dims):
        model.add(GRU(64, input_shape=(None, dims)))

    def two_layer_GRU(dims):
        model.add(GRU(128, input_shape=(None, dims), return_sequences=True))
        model.add(GRU(64))

    def one_layer_LSTM(dims):
        model.add(LSTM(64, input_shape=(None, dims)))

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    MAX_LEN = 71
    embedding = np.load('pca.npy')  # (8985, 1024)
    test = np.load('zh_simplified_test.npy')  # 118615
    train = np.load('zh_simplified_train.npy')  # 355843
    # print(train.shape)
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
    x_train_padded = pad_sequences(x_train, maxlen=10)
    x_test_padded = pad_sequences(x_test, maxlen=10)
    model = Sequential()
    model.add(Embedding(
        8985,
        embedding_dims,
        weights=[embedding],
        input_length=10,
        trainable=False
    ))
    if layer_id == 1:
        one_layer_GRU(embedding_dims)
    if layer_id == 2:
        two_layer_GRU(embedding_dims)
    if layer_id == 3:
        one_layer_LSTM(embedding_dims)
    if layer_id == 4:
        two_layer_LSTM(embedding_dims)
    model.add(Dense(12, input_shape=(None, 32)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', recall_threshold(0)])

    history_result = model.fit(x_train_padded, y_train, batch_size=1024, epochs=50, validation_split=1-training_size,
                               shuffle=True)
    score = model.evaluate(x_test_padded, y_test, batch_size=512)
    # with open('tet.txt', 'w') as f:
    #     pred = model.predict(x_test_padded, batch_size=2048)
    #     output = []
    #     for x in pred:
    #         output.append(np.argmax(output) + 1)
    #     f.write(output)
    print("\ntest_score:{} test_accuracy:{}".format(score[0], score[1]), score)
    # np.save('result/{}_{}_size{}.npy'.format(embedding_dims, layer_id, training_size), score)
    # with open('result/{}_{}_size{}.txt'.format(embedding_dims, layer_id, training_size), 'w') as f:
    #     f.writelines('test_score:{} test_accuracy:{} test_recall:{}'.format(score[0], score[1], score[2]))

if __name__ == '__main__':
 train_evaluate(256, 4, 1)
