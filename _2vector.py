from keras.models import model_from_json, Sequential, Model
from keras.layers import Input
from keras.preprocessing.image import img_to_array, load_img
import numpy
from os import listdir
import pickle
def save_embedding_dims():
    InputImg =[]
    files = listdir('img')
    for file in range(1, 8986):
        InputImg.append(img_to_array(load_img('img/' + '{}.jpg'.format(file))))
    InputImg = numpy.reshape(InputImg, [-1, 3*36*36])/255
    input_img = Input(shape=(3*36*36, ))
    autoencoder = model_from_json(open('net_config.json').read())
    autoencoder.set_weights(numpy.load('weights.npy'))
    encoder = Model(inputs=input_img, outputs=autoencoder.layers[-2](input_img))
    vec_1024 = encoder.predict(InputImg)
    del encoder, autoencoder, InputImg
    # print(vec_1024[1].shape)
    numpy.save('img2vec.npy', vec_1024)
# sequences = []
# lenth = []
# for split in ['test', 'train', 'val']:
#     filename = 'zh_simplified_{}.txt'.format(split)
#     with open(filename, 'r') as f:
#         sequence = []
#         strings = f.readline().split('\t')
#         while strings is not None:
#             strings[2] = strings[2].replace('\n', '')
#             sequence.append(strings[0])
#             sequence.append(strings[1].split(','))
#             sequence.append(len(strings[2]))
#             sequences.append(sequence)
#             lenth.append(len(strings[2]))
# len_max = numpy.max(numpy.array(lenth))
# vec_ = []
# label_ = []
# for x in sequences:
#     temp = []
#     for y in x[1]:
#         temp.append(vec_1024[int(y) + 1])
#         while len(temp) < len_max:
#             temp.append(numpy.zeros(1024,))
#     vec_.append(temp)
#     label_.append(int(x[0]))
# pickle.dump(vec_, 'vec.pkl')
# pickle.dump(label_, 'label.pkl')
