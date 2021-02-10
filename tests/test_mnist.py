from tests.helpers import train, evaluate

import nanograd.nn.module as nn
import nanograd.optim.optimizer as optim
from nanograd.device import Device

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
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 10)
        self.a1 = nn.ReLU()
    
    def forward(self, x):
        x = self.a1(self.bn1(self.l1(x)))
        x = self.l2(x)
        return x.log_softmax()


class CNNModel1d(nn.Module):
    def __init__(self):
        super(CNNModel1d, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.l1 = nn.Linear(3104, 10)
    
    def forward(self, x):
        x = x.reshape(shape=[-1, 1, 784])
        x = self.conv1(x).relu().max_pool1d()
        x = self.conv2(x).relu().max_pool1d()
        x = x.flatten()
        x = self.l1(x)
        return x.log_softmax()


class CNNModel2d(nn.Module):
    def __init__(self):
        super(CNNModel2d, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l1 = nn.Linear(400, 10)
    
    def forward(self, x):
        x = x.reshape(shape=[-1, 1, 28, 28])
        x = self.conv1(x).relu().max_pool2d()
        x = self.conv2(x).relu().max_pool2d()
        x = x.flatten()
        return self.l1(x).log_softmax()


class MNISTTestCPU(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MNISTTestCPU, self).__init__(*args, **kwargs)
        self.device = Device.CPU

    def test_linear(self):
        model = LinearModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train(model, X_train, Y_train, optimizer, steps=1000, device=self.device)
        assert evaluate(model, X_test, Y_test, device=self.device) > 0.95

    def test_conv1d(self):
        model = CNNModel1d()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train(model, X_train, Y_train, optimizer, steps=150, device=self.device)
        assert evaluate(model, X_test, Y_test, device=self.device) > 0.92

    def test_conv2d(self):
        model = CNNModel2d()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train(model, X_train, Y_train, optimizer, steps=200, device=self.device)
        assert evaluate(model, X_test, Y_test, device=self.device) > 0.91


class MNISTTestGPU(MNISTTestCPU):
    def __init__(self, *args, **kwargs):
        super(MNISTTestGPU, self).__init__(*args, **kwargs)
        self.device = Device.GPU