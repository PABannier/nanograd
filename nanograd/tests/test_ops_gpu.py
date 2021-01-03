from tensor import Tensor, Device
from autograd_engine import *
from nn.functional import *

import torch
import numpy as np

from test


def test_add():
    a = Tensor(1, requires_grad=True).to(Device.GPU)
    b = Tensor(2, requires_grad=True).to(Device.GPU)
    c = Tensor(3, requires_grad=True).to(Device.GPU)

    a_torch, b_torch, c_torch = create_identical_torch_tensor(a, b, c)

    d = a + a * b
    d_torch = a_torch + a_torch * b_torch

    e = d + c + Tensor(3)
    e_torch = d_torch + c_torch + torch.tensor(3)

    e.backward()
    e_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)
    check_val_and_grad(d, d_torch)
    check_val_and_grad(e, e_torch)


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
