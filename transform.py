import numpy
import pickle
vec = numpy.load('img2vec.npy')
max_len = 71
temp = []
for split in ['test', 'train', 'val']:
    filename = 'zh_simplified_{}.txt'.format(split)
    with open(filename, 'r') as f:
        sequences = f.readlines()
        sequences_new = []
        for s in sequences:
            s_new = []
            s = s.split('\t')
            s[1] = s[1].split(',')
            s[2] = s[2].replace('\n', '')
            for word_id in s[1]:
                s_new.append(vec[int(int(word_id) - 1)])
            while len(s_new) < max_len:
                s_new.append(numpy.zeros(3*36*36,))
            s[1] = s_new
            sequences_new.append(s)
    numpy.save('zh_simplified_{}.npy'.format(split), sequences_new)

