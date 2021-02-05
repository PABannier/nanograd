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
        self.optimizers = [optim.SGD, optim.Adam, optim.AdamW]
    
    def test_linear_step_classification(self):
        for optimizer in self.optimizers:
                with self.subTest(optimizer=optimizer):
                    simple_model = nnn.Sequential(nnn.Linear(30, 128), nnn.ReLU(), nnn.Linear(128, 10), nnn.ReLU())
                    make_test_step((32, 30), (32, 1), simple_model, optimizer, nnn.NLLLoss(), atol_grad=1e-4, rtol_grad=1e-5)
    
    def test_conv1d_step_classification(self):
        for optimizer in self.optimizers:
            with self.subTest(optimizer=optimizer):
                model = nnn.Sequential(nnn.Conv1d(3, 10, 3, 2), nnn.ReLU(), nnn.MaxPool1d(2), 
                                       nnn.Conv1d(10, 20, 3, 2), nnn.ReLU(), nnn.MaxPool1d(3),
                                       nnn.Flatten(), nnn.Linear(60, 10))
                make_test_step((32, 3, 100), (32, 1), model, optimizer, nnn.NLLLoss(), atol_grad=1e-4, rtol_grad=1e-5)

    def test_conv2d_step_classification(self):
        for optimizer in self.optimizers:
            with self.subTest(optimizer=optimizer):
                model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 2), nnn.ReLU(), nnn.MaxPool2d(3), 
                                       nnn.Conv2d(10, 20, 3, 2), nnn.ReLU(), nnn.MaxPool2d(2), 
                                       nnn.Flatten(), nnn.Linear(80, 10))
                make_test_step((32, 3, 60, 60), (32, 1), model, optimizer, nnn.NLLLoss(), atol=1e-4, rtol=1e-4, atol_grad=5e-4, rtol_grad=1e-4)

    def test_linear_step_regression(self):
        for optimizer in self.optimizers:
                with self.subTest(optimizer=optimizer):
                    simple_model = nnn.Sequential(nnn.Linear(30, 128), nnn.ReLU(), nnn.Linear(128, 1), nnn.ReLU())
                    make_test_step((32, 30), (32, 1), simple_model, optimizer, nnn.MSELoss(), classification=False, atol_grad=1e-4, rtol_grad=1e-5)
    
    def test_conv1d_step_regression(self):
        for optimizer in self.optimizers:
            with self.subTest(optimizer=optimizer):
                model = nnn.Sequential(nnn.Conv1d(3, 10, 3, 2), nnn.ReLU(), nnn.MaxPool1d(2), 
                                       nnn.Conv1d(10, 20, 3, 2), nnn.ReLU(), nnn.MaxPool1d(3),
                                       nnn.Flatten(), nnn.Linear(60, 1))
                make_test_step((32, 3, 100), (32, 1), model, optimizer, nnn.MSELoss(), classification=False, atol_grad=1e-4, rtol_grad=1e-5)

    def test_conv2d_step_regression(self):
        for optimizer in self.optimizers:
            with self.subTest(optimizer=optimizer):
                model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 2), nnn.ReLU(), nnn.MaxPool2d(3), 
                                       nnn.Conv2d(10, 20, 3, 2), nnn.ReLU(), nnn.MaxPool2d(2), 
                                       nnn.Flatten(), nnn.Linear(80, 1))
                make_test_step((32, 3, 60, 60), (32, 1), model, optimizer, nnn.MSELoss(), classification=False, 
                                atol=1e-4, rtol=1e-4, atol_grad=5e-4, rtol_grad=1e-4)