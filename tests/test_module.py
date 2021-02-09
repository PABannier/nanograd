from nanograd.tensor import Tensor
from nanograd.device import Device
import nanograd.nn.module as nnn
import nanograd.optim.optimizer as optim

import unittest

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from tests.helpers import make_test_module


class TestModuleCPU(unittest.TestCase):
    """TestModule encompasses several integration tests to check if the forward 
       and backward passes are correctly carried out. In particular we check the value
       of the output of a model, and the gradients of all the weights and biases of that model.

       Those tests are designed to check that the gradient loss correctly backpropagates throughout
       the network. 

       ..note: The numerical schema seems unstable for some modules namely, MaxPool1d, MaxPool2d, AvgPool1d,
               AvgPool2d, BacthNorm1d, BatchNorm2d.
    """
    def __init__(self, *args, **kwargs):
        super(TestModuleCPU, self).__init__(*args, **kwargs)
        self.device = Device.CPU
    
    def test_linear(self):
            model = nnn.Sequential(nnn.Linear(256, 128), nnn.LeakyReLU(), nnn.Linear(128, 1), nnn.ReLU())
            make_test_module((8, 256), (8, 1), model, device=self.device)
    
    def test_batchnorm1d(self):
        for num_features_1 in [16, 32, 64, 128, 256]:
            for num_features_2 in [16, 32, 64, 128, 256]:
                with self.subTest(num_features_1=num_features_1, num_features_2=num_features_2):
                    model = nnn.Sequential(nnn.Linear(256, num_features_1), nnn.BatchNorm1d(num_features_1), 
                                           nnn.Linear(num_features_1, num_features_2), nnn.BatchNorm1d(num_features_2), 
                                           nnn.LeakyReLU(), nnn.Linear(num_features_2, 1))
                    make_test_module((8, 256), (8, 1), model, device=self.device, atol_grad=1e-4)
    
    def test_conv1d(self):
        for kernel_size in range(2, 5):
            for stride in range(1, 4):
                for padding in range(0):
                    with self.subTest(kernel_size=kernel_size, stride=stride, padding=padding):
                        model = nnn.Sequential(nnn.Conv1d(4, 32, kernel_size, stride, padding), nnn.MaxPool1d(2, 2), nnn.ReLU(),
                                               nnn.Conv1d(32, 64, kernel_size, stride, padding), nnn.AvgPool1d(2, 2), nnn.Flatten())
                        make_test_module((8, 4, 100), (8, 1), model, device=self.device)

    def test_maxpool1d(self):
        for kernel_size in range(2, 5):
                with self.subTest(kernel_size=kernel_size):
                    model = nnn.Sequential(nnn.Conv1d(4, 32, 3, 2), nnn.MaxPool1d(kernel_size), nnn.ReLU(),
                                           nnn.Conv1d(32, 64, 3, 2), nnn.MaxPool1d(kernel_size), nnn.Flatten())
                    make_test_module((8, 4, 100), (8, 1), model, device=self.device, atol_grad=1e-4)
    
    def test_avgpool1d(self):
        for kernel_size in range(2, 5):
                with self.subTest(kernel_size=kernel_size):
                    model = nnn.Sequential(nnn.Conv1d(4, 32, 3, 2), nnn.AvgPool1d(kernel_size), nnn.ReLU(),
                                           nnn.Conv1d(32, 64, 3, 2), nnn.AvgPool1d(kernel_size), nnn.Flatten())
                    make_test_module((8, 4, 100), (8, 1), model, device=self.device, atol_grad=1e-4)

    def test_conv2d(self):
        for kernel_size in range(2, 5):
            for stride in range(1, 4):
                for padding in range(0):
                    with self.subTest(kernel_size=kernel_size, stride=stride, padding=padding):
                        model = nnn.Sequential(nnn.Conv2d(3, 32, kernel_size, stride, padding), nnn.MaxPool2d(2), nnn.ReLU(),
                                               nnn.Conv2d(32, 64, kernel_size, stride, padding), nnn.AvgPool2d(2), nnn.ReLU(),
                                               nnn.Flatten())
                        make_test_module((8, 3, 80, 80), (8, 1), model, device=self.device)
    
    def test_maxpool2d(self):
        for kernel_size in range(2, 5):
            with self.subTest(kernel_size=kernel_size):
                model = nnn.Sequential(nnn.Conv2d(3, 32, 3, 2), nnn.MaxPool2d(kernel_size), nnn.ReLU(),
                                       nnn.Conv2d(32, 64, 3, 2), nnn.MaxPool2d(kernel_size), nnn.ReLU(),
                                       nnn.Flatten())
                make_test_module((8, 3, 80, 80), (8, 1), model, device=self.device, atol_grad=1e-4)
    
    def test_avgpool2d(self):
        for kernel_size in range(2, 5):
            with self.subTest(kernel_size=kernel_size):
                model = nnn.Sequential(nnn.Conv2d(3, 32, 3, 2), nnn.AvgPool2d(kernel_size), nnn.ReLU(),
                                       nnn.Conv2d(32, 64, 3, 2), nnn.AvgPool2d(kernel_size), nnn.ReLU(),
                                       nnn.Flatten())
                make_test_module((8, 3, 80, 80), (8, 1), model, device=self.device, atol_grad=1e-4, rtol_grad=1e-5)
    
    def test_batchnorm2d(self):
        for num_channels_1 in [8, 16]:  
            for num_channels_2 in [16, 32]: 
                if num_channels_1 < num_channels_2:
                    with self.subTest(num_channels_1=num_channels_1, num_channels_2=num_channels_2):
                        model = nnn.Sequential(nnn.Conv2d(3, num_channels_1, 2, 2), nnn.BatchNorm2d(num_channels_1), nnn.ReLU(),
                                            nnn.Conv2d(num_channels_1, num_channels_2, 3, 3), nnn.BatchNorm2d(num_channels_2), 
                                            nnn.ReLU(), nnn.Flatten())
                        make_test_module((8, 3, 30, 30), (8, 1), model, device=self.device, atol_grad=5e-4, rtol_grad=1e-5)


class TestModuleGPU(TestModuleCPU):
    def __init__(self, *args, **kwargs):
        super(TestModuleGPU, self).__init__(*args, **kwargs)
        self.device = Device.GPU