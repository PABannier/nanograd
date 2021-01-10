from tensor import Tensor, Device
from autograd_engine import *
from nn.functional import *

import torch
import numpy as np

from tests.helpers import *


def test_add_forward():
    a = Tensor.normal(0, 1, (3, 3, 3))
    b = Tensor.normal(0, 1, (3, 3, 1))
    c = Tensor.normal(0, 1, (1, 3, 3))

    a_torch, b_torch, c_torch = create_identical_torch_tensor(a, b, c)

    a.to(Device.GPU)
    b.to(Device.GPU)
    c.to(Device.GPU)

    d = a + b
    d_torch = a_torch + b_torch

    e = d + c 
    e_torch = d_torch + c_torch

    f = e + c + a + b
    f_torch = e_torch + c_torch + a_torch + b_torch

    d.to(Device.CPU)
    e.to(Device.CPU)
    f.to(Device.CPU)

    check_val(d, d_torch)
    check_val(e, e_torch)
    check_val(f, f_torch)

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



"""
def test_neg():
    a = Tensor.randn(30, 40, requires_grad=True).to(Device.GPU)
    a_torch = create_identical_torch_tensor(a)

    b = -a
    b_torch = -a_torch

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_mul():
    a = Tensor.randn(30, 40, requires_grad=True).to(Device.GPU)
    b = Tensor.randn(30, 40, requires_grad=True).to(Device.GPU)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a * b
    c_torch = a_torch * b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)


def test_div():
    a = Tensor.randn(30, 40, requires_grad=True).to(Device.GPU)
    b = Tensor.randn(30, 40, requires_grad=True).to(Device.GPU)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a / b
    c_torch = a_torch / b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)
"""