from nanograd.tensor import Tensor
from nanograd.device import Device

import torch
import torch.nn.functional as F
import numpy as np

import unittest

from tests.helpers import make_test_ops


class TestOps(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOps, self).__init__(*args, **kwargs)
        self.device = Device.CPU

    def test_add(self):
        make_test_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x+y, device=self.device, name='add')
    def test_sub(self):
        make_test_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x-y, device=self.device, name='sub')
    def test_mul(self):
        make_test_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x*y, device=self.device, name='mul')
    def test_div(self):
        make_test_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x/y, device=self.device, name='div')
    def test_matmul(self):
        make_test_ops([(3, 4), (4, 2)], lambda x,y: x@y, device=self.device, name='matmul')
    def test_rsub(self):
        make_test_ops([(5, 4, 3)], lambda x: 3 - x, device=self.device, name='rsub')
    def test_rmul(self):
        make_test_ops([(5, 4, 3)], lambda x: 6 * x, device=self.device, name='rmul')
    def test_radd(self):
        make_test_ops([(5, 4, 3)], lambda x: 3 + x, device=self.device, name='radd')
    
    def test_add_broadcast(self):
        make_test_ops([(10, 10), (10, 1)], lambda x,y: x+y, device=self.device, name='add_broadcast')
        make_test_ops([(20, 30, 30), (20, 1, 1)], lambda x,y: x+y, device=self.device, name='add_broadcast')
    def test_sub_broadcast(self):
        make_test_ops([(10, 10), (10, 1)], lambda x,y: x-y, device=self.device, name='sub_broadcast')
        make_test_ops([(20, 30, 30), (20, 1, 1)], lambda x,y: x-y, device=self.device, name='sub_broadcast')
    def test_mul_broadcast(self):
        make_test_ops([(10, 10), (10, 1)], lambda x,y: x*y, device=self.device, name='mul_broadcast')
        make_test_ops([(20, 30, 30), (20, 1, 1)], lambda x,y: x*y, device=self.device, name='mul_broadcast')
    def test_div_broadcast(self):
        make_test_ops([(10, 10), (10, 1)], lambda x,y: x/y, device=self.device, name='div_broadcast')
        make_test_ops([(20, 30, 30), (20, 1, 1)], lambda x,y: x/y, device=self.device, name='div_broadcast')
    
    def test_neg(self):
        make_test_ops([(10, 15, 3)], lambda x: -x, device=self.device, name='neg_broadcast')
    def test_pow(self):
        make_test_ops([(20, 20)], lambda x: x ** 3, device=self.device, name='pow_broadcast')
        make_test_ops([(20, 20)], lambda x: x ** -2, device=self.device, name='pow_broadcast')
        make_test_ops([(20, 20)], lambda x: x ** 0.3, device=self.device, name='pow_broadcast')
    
    def test_log(self):
        make_test_ops([(20, 20)], lambda x: x.log(), device=self.device, name='log')
    def test_exp(self):
        make_test_ops([(20, 20)], lambda x: x.exp(), device=self.device, name='exp')
    def test_sqrt(self):
        make_test_ops([(20, 20)], lambda x: x.sqrt(), device=self.device, name='sqrt')
    
    def test_reshape(self):
        make_test_ops([(20, 20)], lambda x: x.reshape(shape=[400, 1]), device=self.device, name='reshape')
        make_test_ops([(256, 2)], lambda x: x.reshape(shape=[8, 8, 8]), device=self.device, name='reshape')
    def test_transpose(self):
        make_test_ops([(20, 10)], lambda x: x.T, device=self.device, name='transpose')
    
    def test_sum(self):
        make_test_ops([(20, 15)], lambda x: x.sum(), device=self.device, name='sum')
        make_test_ops([(20, 15, 3)], lambda x: x.sum(axis=1), device=self.device, name='sum')
        make_test_ops([(30, 40, 20)], lambda x: x.sum(axis=(1, 2)), device=self.device, name='sum')
    def test_mean(self):
        make_test_ops([(20, 20)], lambda x: x.mean(axis=0), device=self.device, name='mean')
        make_test_ops([(20, 20)], lambda x: x.mean(axis=1), device=self.device, name='mean')
    def test_min(self):
        make_test_ops([(30, 40, 20, 10)], lambda x: x.min(axis=3).min(axis=0), lambda x: x.min(axis=3)[0].min(axis=0)[0], device=self.device, name='min')
    def test_max(self):
        make_test_ops([(30, 40, 20, 10)], lambda x: x.max(axis=3).max(axis=0), lambda x: x.max(axis=3)[0].max(axis=0)[0], device=self.device, name='max')
    def test_slice(self):
        make_test_ops([(30, 40, 20, 10)], lambda x: x[10:20, 3:, :, :], device=self.device, name='slice')
        make_test_ops([(30, 40, 20, 10)], lambda x: x[10:20, :5, :12, 5:], device=self.device, name='slice')
        make_test_ops([(30, 40, 20, 10)], lambda x: x[10:20, 4:5, :12, 0:-1], device=self.device, name='slice')
    def test_squeeze(self):
        make_test_ops([(30, 1)], lambda x: x.squeeze(axis=1), device=self.device, name='squeeze')
        make_test_ops([(1, 3)], lambda x: x.squeeze(axis=0), device=self.device, name='squeeze')
    def test_unsqueeze(self):
        make_test_ops([(30, )], lambda x: x.unsqueeze(axis=1), device=self.device, name='unsqueeze')
    def test_one_hot(self):
        make_test_ops([(100, )], lambda x: x.onehot(num_classes=10), 
                      lambda x: F.one_hot(x.long(), 10), test_backward=False, discrete=True, device=self.device, name='one_hot')
    
    def test_relu(self):
        make_test_ops([(20, 20)], lambda x: x.relu(), device=self.device, name='relu')
    def test_tanh(self):
        make_test_ops([(20, 20)], lambda x: x.tanh(), device=self.device, name='tanh')
    def test_sigmoid(self):
        make_test_ops([(20, 20)], lambda x: x.sigmoid(), device=self.device, name='sigmoid')
    def test_logsoftmax(self):
        make_test_ops([(128, 10)], lambda x: x.log_softmax(), 
                      lambda x: F.log_softmax(x, dim=1), device=self.device, name='logsoftmax')
    
    def test_pad1d(self):
        for pad_size in range(5):
            with self.subTest(pad_size=pad_size):
                make_test_ops([(20, 3, 100)], lambda x: x.pad1d((pad_size, pad_size)), 
                              lambda x: F.pad(x, (pad_size, pad_size)), device=self.device, name='pad1d')
    def test_maxpool1d(self):
        for pool_size in range(1, 4):
            with self.subTest(pool_size=pool_size):
                make_test_ops([(20, 3, 100)], lambda x: x.max_pool1d(pool_size), 
                              lambda x: F.max_pool1d(x, pool_size), device=self.device, name='maxpool1d')
    def test_avgpool1d(self):
        for pool_size in range(1, 4):
            with self.subTest(pool_size=pool_size):
                make_test_ops([(20, 3, 100)], lambda x: x.avg_pool1d(pool_size), 
                              lambda x: F.avg_pool1d(x, pool_size), device=self.device, name='avgpool1d')
    def test_conv1d(self):
        for kernel_size in range(2, 6):
            for stride in range(1, 5):
                with self.subTest(kernel_size=kernel_size, stride=stride):
                    make_test_ops([(20, 3, 100), (10, 3, kernel_size)], lambda x, w: x.conv1d(w, stride=stride), 
                                  lambda x, w: F.conv1d(x, w, stride=stride), device=self.device, name='conv1d')
    
    def test_pad2d(self):
        make_test_ops([(20, 3, 30, 30)], lambda x: x.pad2d((1, 2, 3, 4)), 
                      lambda x: F.pad(x, (1, 2, 3, 4)), device=self.device, name='pad2d')
    def test_maxpool2d(self):
        for pool_size in range(1, 5):
            with self.subTest(pool_size=pool_size):
                make_test_ops([(20, 3, 30, 30)], lambda x: x.max_pool2d((pool_size, pool_size)), 
                              lambda x: F.max_pool2d(x, kernel_size=(pool_size, pool_size)), device=self.device, name='maxpool2d')
    def test_avgpool2d(self):
        for pool_size in range(1, 5):
            with self.subTest(pool_size=pool_size):
                make_test_ops([(20, 3, 30, 30)], lambda x: x.avg_pool2d((pool_size, pool_size)), 
                              lambda x: F.avg_pool2d(x, kernel_size=(pool_size, pool_size)), device=self.device, name='avgpool2d')
    def test_conv2d(self):
        for kernel_size in range(2, 6):
            for stride in range(1, 5):
                with self.subTest(kernel_size=kernel_size, stride=stride):
                    make_test_ops([(20, 3, 30, 30), (10, 3, kernel_size, kernel_size)], lambda x, w: x.conv2d(w, stride=stride), 
                                  lambda x, w: F.conv2d(x, w, stride=stride), device=self.device, name='conv2d')

class TestGPU(TestOps):
    def __init__(self, *args, **kwargs):
        super(TestGPU, self).__init__(*args, **kwargs)
        self.device = Device.GPU