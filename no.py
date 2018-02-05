import numpy as np
from numpy import linalg as la

pre = np.load('img2vec.npy')
t1 = pre[433]
t2 = pre[4277]


def cosSimilar(inA, inB):
    inA = np.mat(inA)
    inB = np.mat(inB)
    num = float(inA * inB.T)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

print(cosSimilar(t1, t2))
