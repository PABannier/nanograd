from tests.helpers import train, evaluate

import nanograd.nn.module as nn
import nanograd.optim.optimizer as optim

import numpy as np

import unittest
import gzip

MNIST_PATHS = [
    'data/train-images-idx3-ubyte.gz',
    'data/train-labels-idx1-ubyte.gz',
    'data/t10k-images-idx3-ubyte.gz',
    'data/t10k-labels-idx1-ubyte.gz'
]


def load_mnist():
    mnist = []
    for path in MNIST_PATHS:
        with open(path, 'rb') as f:
            dat = f.read()
            arr = np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
            mnist.append(arr)
    
    X_train, Y_train, X_test, Y_test = tuple(mnist)

    X_train = X_train[0x10:].reshape((-1, 784)).astype(np.float32)
    Y_train = Y_train[8:]
    X_test = X_test[0x10:].reshape((-1, 784)).astype(np.float32)
    Y_test = Y_test[8:]

    X_train = np.divide(X_train, 255.0)
    X_test = np.divide(X_test, 255.0)
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_mnist()


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 10)
        self.a1 = nn.ReLU()
    
    def forward(self, x):
        x = self.a1(self.bn1(self.l1(x)))
        x = self.l2(x)
        return x.log_softmax()


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.l1 = nn.Linear(400, 10)
    
    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))
        x = self.bn1(self.conv1(x)).relu().max_pool2d()
        x = self.conv2(x).relu().max_pool2d()
        x = x.flatten()
        return self.l1(x).log_softmax()


class MNISTTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MNISTTest, self).__init__(*args, **kwargs)
        self.optimizer = [optim.Adam, optim.AdamW]
    
    def test_linear(self):
        for optimizer in self.optimizer:
            with self.subTest(optimizer=optimizer):
                model = LinearModel()
                optimizer = optimizer(model.parameters(), lr=1e-3)
                train(model, X_train, Y_train, optimizer, steps=1000)
                assert evaluate(model, X_test, Y_test) > 0.95

    def test_conv(self):
        for optimizer in self.optimizer:
            with self.subTest(optimizer=optimizer):
                model = CNNModel()
                optimizer = optimizer(model.parameters(), lr=1e-3)
                train(model, X_train, Y_train, optimizer, steps=200)
                assert evaluate(model, X_test, Y_test) > 0.92