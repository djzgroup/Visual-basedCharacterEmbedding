import numpy as np
import numexpr as ne
from boto.ec2.autoscale import request
from gensim.models import Word2Vec
from keras.layers import LSTM, Embedding, Dense, Dropout, Activation, GRU, Merge
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import *
from matplotlib import pyplot

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
def run(dim, layer, train_size, padding_len):
    def one_hot(x_):
        index = 0
        temp = np.zeros([x_.shape[0], 12])
        for item in x_:
            temp[index][int(item) - 1] = 1
            index += 1
        return temp
    def two_layer_LSTM(dims):
        model_word2vec.add(LSTM(256, input_shape=(None, dims), activation='relu', return_sequences=True))
        model_word2vec.add(LSTM(128, return_sequences=True))
        model_word2vec.add(LSTM(64))
        # model_word2vec.add(BatchNormalization(axis=1))

    def one_layer_GRU(dims):
        model_word2vec.add(GRU(64, input_shape=(None, dims), activation='relu'))
        # model_word2vec.add(BatchNormalization(axis=1))

    def two_layer_GRU(dims):
        model_word2vec.add(GRU(128, input_shape=(None, dims), return_sequences=True))
        model_word2vec.add(GRU(64))

    def one_layer_LSTM(dims):
        model_word2vec.add(LSTM(64, input_shape=(None, dims)))

    bin_fname = 'word2vec/word2vec_wx'
    model_word2vec = Word2Vec.load(bin_fname)
    EMBEDDING_DIMS = 256
    d = {}
    for split in ['train', 'test', 'val']:
        x = []
        y = []
        src_name = 'data/zh_simplified_{}.txt'.format(split)
        with open(src_name, 'r', encoding='utf-8') as f:
            sequence = f.readline()
            while sequence is not '':
                New_seq = []
                sequence = sequence.split('\t')
                sequence[1] = sequence[1].replace('\n', '')
                for char in sequence[1]:
                    # New_seq.append(model[char])
                    if char not in d.keys():
                        d[char] = len(d.keys())
                x.append(New_seq)
                y.append(sequence[0])
                sequence = f.readline()
    for split in ['train', 'test', 'val']:
        x = []
        y = []
        src_name = 'data/zh_simplified_{}.txt'.format(split)
        with open(src_name, 'r', encoding='utf-8') as f:
            sequence = f.readline()
            while sequence is not '':
                New_seq = []
                sequence = sequence.split('\t')
                sequence[1] = sequence[1].replace('\n', '')
                for char in sequence[1]:
                    New_seq.append(d[char])
                x.append(New_seq)
                y.append(sequence[0])
                sequence = f.readline()
        np.save('zh_simplified_word2vec_{}_{}.npy'.format(split, 'data'), x)
        np.save('zh_simplified_word2vec_{}_{}.npy'.format(split, 'label'), y)
    embedding_matrix = np.zeros((len(d) + 1, EMBEDDING_DIMS))
    model_word2vec = model_word2vec.wv
    for word, i in d.items():
        if word in model_word2vec:
            embedding_matrix[i] = model_word2vec[word]
    del d, model_word2vec
    x_train = np.load('zh_simplified_word2vec_train_data.npy')
    y_train = np.load('zh_simplified_word2vec_train_label.npy')
    x_ = np.load('zh_simplified_word2vec_test_data.npy')
    y_ = np.load('zh_simplified_word2vec_test_label.npy')
    x_test = np.load('zh_simplified_word2vec_val_data.npy')
    y_test = np.load('zh_simplified_word2vec_val_label.npy')
    x_train = np.append(x_train, x_)
    y_train = np.append(y_train, y_)
    embedding = np.load('pca.npy')
    x_train_padded = pad_sequences(x_train, maxlen=padding_len)
    x_padded = pad_sequences(x_, maxlen=padding_len)
    y_train = one_hot(y_train)
    y_= one_hot(y_)
    embedding_layer_img2vec = Embedding(
        8985,
        256,
        weights=[embedding],
        input_length=padding_len,
        trainable=False
    )
    embedding_layer_word2vec = Embedding(
        8986,
        256,
        weights=[embedding_matrix],
        input_length=padding_len,
        trainable=True
    )
    model_word2vec = Sequential()
    model_word2vec.add(embedding_layer_word2vec)
    if layer == 1:
        one_layer_GRU(dim)
    if layer == 2:
        two_layer_GRU(dim)
    if layer == 3:
        one_layer_LSTM(dim)
    if layer == 4:
        two_layer_LSTM(dim)

    model_img2vec = Sequential()
    model_img2vec.add(embedding_layer_img2vec)
    model_img2vec.add(BatchNormalization(axis=1))
    model_img2vec.add(LSTM(256, input_shape=(None, dims), activation='tanh', return_sequences=True))
    model_img2vec.add(LSTM(128, activation='tanh', return_sequences=True))
    model_img2vec.add(LSTM(64))
    # merged = Merge([model_word2vec, model_img2vec], mode='concat')

    model = Sequential()
    model.add(model_img2vec)
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy', f1score])
    print(model_img2vec.summary())
    history = model.fit(x_train_padded, y_train, batch_size=128, epochs=50, validation_split=1 - train_size,
              shuffle=True)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    # pyplot.savefig("{}.png".format(dim))
    score = model.evaluate(x_padded, y_, batch_size=128)
    print("\ntest_score:{} test_accuracy:{}".format(score[0], score[1]), score)
    np.save('result/word2vec_{}_{}_len{}.npy'.format(dim, layer, padding_len), score)
    with open('result/word2vec_{}_{}_len{}.txt'.format(dim, layer, padding_len), 'w') as f:
        f.writelines('test_score:{} test_accuracy:{} test_recall:{}'.format(score[0], score[1], score[2]))


if __name__ == '__main__':
    layers = [4]
    train_size = [1]
    dims = [256]
    lens = [10]
    for len_ in lens:
        for dim in dims:
            for layer in layers:
                for size in train_size:
                    run(dim, layer, train_size=0.75, padding_len=len_)
