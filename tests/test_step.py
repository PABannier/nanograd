from nanograd.tensor import Tensor
from nanograd.device import Device

import nanograd.nn.module as nnn
import nanograd.optim.optimizer as optim

import torch
import torch.nn.functional as F
import torch.optim

from tests.helpers import make_test_step
import numpy as np
import unittest

torch.set_printoptions(precision=8)


x_init = np.random.randn(1, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


def step_nanograd(optim, kwargs={}):
    net = TinyNet().train()
    optim = optim([net.x, net.W], **kwargs)
    out = net.forward()
    out.backward()
    optim.step()

    return net.x.cpu().data, net.W.cpu().data

def step_pytorch(optim, kwargs={}):
    net = TorchNet()
    optim = optim([net.x, net.W], **kwargs)
    out = net.forward()
    out.backward()
    optim.step()

    return net.x.detach().numpy(), net.W.detach().numpy()


class TinyNet(nnn.Module):
    def __init__(self):
        super().__init__()
        self.x = Tensor(x_init.copy(), requires_grad=True)
        self.W = Tensor(W_init.copy(), requires_grad=True)
        self.m = Tensor(m_init.copy())

    def forward(self):
        out = (self.x @ self.W).relu()
        out = out.log_softmax()
        out = out.__mul__(self.m).__add__(self.m).sum()
        return out


class TorchNet():
    def __init__(self):
        self.x = torch.tensor(x_init.copy(), requires_grad=True)
        self.W = torch.tensor(W_init.copy(), requires_grad=True)
        self.m = torch.tensor(m_init.copy())

    def forward(self):
        out = (self.x @ self.W).relu()
        out = F.log_softmax(out, 1)
        out = out.__mul__(self.m).__add__(self.m).sum()
        return out


class TestStep(unittest.TestCase):
    def test_sgd(self):
        for mom in [0, 0.9]:
            with self.subTest(mom=mom):
                kwargs = {'lr': 0.001, 'momentum': mom}
                for x, y in zip(step_nanograd(optim.SGD, kwargs), step_pytorch(torch.optim.SGD, kwargs)):
                    np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_adam(self):
        for x, y in zip(step_nanograd(optim.Adam), step_pytorch(torch.optim.Adam)):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_adamw(self):
        for wd in [1e-1, 1e-2, 1e-3]:
            with self.subTest(wd=wd):
                kwargs = {'lr': 1e-3, 'weight_decay': wd}
                for x, y in zip(step_nanograd(optim.AdamW, kwargs), step_pytorch(torch.optim.AdamW, kwargs)):
                    np.testing.assert_allclose(x, y, atol=1e-5)
    
