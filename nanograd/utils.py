import numpy as np
import gzip


MNIST_PATHS = [
    '../data/train-images-idx3-ubyte.gz',
    '../data/train-labels-idx1-ubyte.gz',
    '../data/t10k-images-idx3-ubyte.gz',
    '../data/t10k-labels-idx1-ubyte.gz'
]


def load_mnist():
    print("Loading data...")
    mnist = []
    for path in MNIST_PATHS:
        with open(path, 'rb') as f:
            dat = f.read()
            arr = np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
            mnist.append(arr)
    
    return tuple(mnist)