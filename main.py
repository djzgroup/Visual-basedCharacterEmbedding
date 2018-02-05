from build_dataset import build_embedding_dims_autoencoder
from _2vector import save_embedding_dims
from network import train_evaluate
layers = [4]
train_size = [0.5]
# 1 means one layer GRU
# 2 means two layers GRU
# 3 means one layer LSTM
# 4 means two layers LSTM
if __name__ == '__main__':
    dims = [128]
    for dim in dims:
        # build_embedding_dims_autoencoder(dim)
        # save_embedding_dims()
        for layer in layers:
            for size in train_size:
                train_evaluate(dim, layer, training_size=size)
