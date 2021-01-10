from tensor import Tensor, Device
from autograd_engine import *
from nn.functional import *

import torch
import numpy as np

from tests.helpers import *


def test_add_forward():
    a = Tensor.normal(0, 1, (3, 3, 3))
    b = Tensor.normal(0, 1, (3, 3, 1))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.to(Device.GPU)
    b.to(Device.GPU)

    d = a + b
    d_torch = a_torch + b_torch

    d.to(Device.CPU)

    check_val(d, d_torch)

def test_neg_forward():
    a = Tensor.normal(0, 1, (8, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.to(Device.GPU)

    b = -a
    b_torch = -a_torch
    b.to(Device.CPU)

    check_val(b, b_torch)

def test_sub_forward():
    a = Tensor.normal(0, 1, (3, 3, 3))
    b = Tensor.normal(0, 1, (3, 3, 1))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.to(Device.GPU)
    b.to(Device.GPU)

    d = a - b
    d_torch = a_torch - b_torch

    d.to(Device.CPU)

    check_val(d, d_torch)

def test_mul_forward():
    a = Tensor.normal(30, 2, (3, 2, 1))
    b = Tensor.normal(30, 2, (3, 2, 3))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.to(Device.GPU)
    b.to(Device.GPU)

    c = a * b
    c_torch = a_torch * b_torch

    c.to(Device.CPU)

    check_val(c, c_torch)

def test_log_forward():
    a = Tensor.normal(30, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.to(Device.GPU)

    b = a.log()
    b_torch = a_torch.log()
    b.to(Device.CPU)

    check_val(b, b_torch)

def test_exp_forward():
    a = Tensor.normal(4, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.to(Device.GPU)

    b = a.exp()
    b_torch = a_torch.exp()
    b.to(Device.CPU)

    check_val(b, b_torch)

def test_pow_forward():
    a = Tensor.normal(4, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.to(Device.GPU)

    b = a ** 3.0
    b_torch = a_torch ** 3.0

    c = a ** -1.0
    c_torch = a_torch ** -1.0

    d = a ** 1.5
    d_torch = a_torch ** 1.5

    b.to(Device.CPU)
    c.to(Device.CPU)
    d.to(Device.CPU)

    check_val(b, b_torch)
    check_val(c, c_torch)
    check_val(d, d_torch)

def test_div_forward():
    a = Tensor.normal(30, 2, (3, 2, 1))
    b = Tensor.normal(30, 2, (3, 2, 3))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.to(Device.GPU)
    b.to(Device.GPU)

    c = a / b
    c_torch = a_torch / b_torch

    c.to(Device.CPU)

    check_val(c, c_torch, atol=1e-3)

def test_relu_forward():
    a = Tensor.normal(0, 2, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.to(Device.GPU)

    b = a.relu()
    b_torch = a_torch.relu()

    b.to(Device.CPU)

    check_val(b, b_torch)

def test_sigmoid_forward(): 
    a = Tensor.randn(20, 20)
    a_torch = create_identical_torch_tensor(a)

    a.to(Device.GPU)

    b = a.sigmoid()
    b_torch = a_torch.sigmoid()
    b.to(Device.CPU)

    check_val(b, b_torch)

def test_tanh_forward():
    a = Tensor.randn(20, 20)
    a_torch = create_identical_torch_tensor(a)

    a.to(Device.GPU)

    b = a.tanh()
    b_torch = a_torch.tanh()
    b.to(Device.CPU)

    check_val(b, b_torch)