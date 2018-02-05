from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
bin_fname = 'word2vec/word2vec_wx'
model = Word2Vec.load(bin_fname)
model = model.wv
img_embedding = np.load('img2vec.npy')
def show(martix):
    a = np.reshape(martix, (16,16))
    plt.imshow(a)
from PIL import Image
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
img = []
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
for i in range(8984):
    img.append(ImageToMatrix('G:\Yeah!!!\img2vec/img/{}.jpg'.format(i + 1)))
img = np.array(img)
img = np.reshape(img, (8984, -1))
pca_model = PCA(10)
y = pca_model.fit_transform(img)
np.set_printoptions(precision=9)
print(y[178])