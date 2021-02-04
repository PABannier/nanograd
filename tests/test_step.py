from nanograd.tensor import Tensor
from nanograd.device import Device
import nanograd.nn.module as nnn
import nanograd.optim.optimizer as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import unittest

from tests.helpers import make_test_step


class TestStep(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestStep, self).__init__(*args, **kwargs)
        self.device = Device.CPU
        self.optimizers = [optim.SGD] # optim.Adam, optim.AdamW
        self.criteria = [nnn.NLLLoss()]
    
    def test_linear_step(self):
        for optimizer in self.optimizers:
            for criterion in self.criteria:
                with self.subTest(optimizer=optimizer, criterion=criterion):
                    make_test_step((32, 30), (32, 1), SimpleLinearModel(), SimpleTorchLinearModel(), optimizer, criterion)
    
    """
    def test_conv1d_step(self):
        for optimizer in self.optimizers:
            for criterion in self.criteria:
                with self.subTest(optimizer=optimizer, criterion=criterion):
                    model, model_torch = CNN1dModel(), CNN1dTorchModel()
                    make_test_step((32, 3, 100), (32, 1), model, model_torch, optimizer, criterion)
    
    def test_conv2d_step(self):
        for optimizer in self.optimizers:
            for criterion in self.criteria:
                with self.subTest(optimizer=optimizer, criterion=criterion):
                    model, model_torch = CNN2dModel(), CNN2dTorchModel()
                    make_test_step((32, 3, 60, 60), (32, 1), model, model_torch, optimizer, criterion)
    """

    
class SimpleLinearModel(nnn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nnn.Linear(30, 128), nnn.Linear(128, 10)
        self.a1, self.a2 = nnn.ReLU(), nnn.ReLU()
    
    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return x, x.log_softmax()


class SimpleTorchLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(30, 128), nn.Linear(128, 10)
        self.a1, self.a2 = nn.ReLU(), nn.ReLU()
    
    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return x, F.log_softmax(x, dim=1)


class CNN1dModel(nnn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.bn1 = nnn.Conv1d(3, 10, 3, 2), nnn.BatchNorm1d(10)
        self.c2, self.bn2 = nnn.Conv1d(10, 20, 3, 2), nnn.BatchNorm1d(20)
        self.l1 = nnn.Linear(1600, 10)
    
    def forward(self, x):
        x = self.bn1(self.c1(x).relu())
        x = self.bn2(self.c2(x).relu())
        x = nnn.Flatten()(x)
        x = self.l1(x)
        return x.log_softmax()


class CNN1dTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.bn1 = nn.Conv1d(3, 10, 3, 2), nn.BatchNorm1d(10)
        self.c2, self.bn2 = nn.Conv1d(10, 20, 3, 2), nn.BatchNorm1d(20)
        self.l1 = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.bn1(self.c1(x).relu())
        x = self.bn2(self.c2(x).relu())
        x = nn.Flatten()(x)
        x = self.l1(x)
        return F.log_softmax(x, dim=1)


class CNN2dModel(nnn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.bn1 = nnn.Conv2d(3, 10, 3, 2), nnn.BatchNorm2d(10)
        self.c2, self.bn2 = nnn.Conv2d(10, 20, 3, 2), nnn.BatchNorm2d(20)
        self.l1 = nnn.Linear(3920, 10)
    
    def forward(self, x):
        x = self.bn1(self.c1(x).relu())
        x = self.bn2(self.c2(x).relu())
        x = nnn.Flatten()(x)
        x = self.l1(x)
        return x.log_softmax()


class CNN2dTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.bn1 = nn.Conv2d(3, 10, 3, 2), nn.BatchNorm2d(10)
        self.c2, self.bn2 = nn.Conv2d(10, 20, 3, 2), nn.BatchNorm2d(20)
        self.l1 = nn.Linear(3920, 10)

    def forward(self, x):
        x = self.bn1(self.c1(x).relu())
        x = self.bn2(self.c2(x).relu())
        x = nn.Flatten()(x)
        x = self.l1(x)
        return F.log_softmax(x, dim=1)