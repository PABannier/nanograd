from nanograd.tensor import Tensor
from nanograd.device import Device

import torch
import numpy as np

from tests.helpers import check_val_and_grad, create_identical_torch_tensor

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def test_add():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    b = Tensor.normal(0, 1, (3, 3, 1), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    c = a + b
    c_torch = a_torch + b_torch

    c.backward()
    c_torch.sum().backward()

    c.cpu(), b.cpu(), a.cpu()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_add_broadcast():
    a = Tensor.normal(0, 1, (20, 30), requires_grad=True)
    b = Tensor.normal(0, 1, (20, 1), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu(), b.gpu()

    c = a + b
    c_torch = a_torch + b_torch

    c.backward()
    c_torch.sum().backward()

    a.cpu(), b.cpu(), c.cpu()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_neg():
    a = Tensor.normal(0, 1, (8, 10, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = -a
    b_torch = -a_torch

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sub():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    b = Tensor.normal(0, 1, (3, 3, 1), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    d = a - b
    d_torch = a_torch - b_torch

    d.backward()
    d_torch.sum().backward()

    a.cpu(), b.cpu(), d.cpu()

    check_val_and_grad(d, d_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_mul():
    a = Tensor.normal(30, 2, (3, 2, 1), requires_grad=True)
    b = Tensor.normal(30, 2, (3, 2, 3), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    c = a * b
    c_torch = a_torch * b_torch

    c.backward()
    c_torch.sum().backward()

    c.cpu(), b.cpu(), a.cpu()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_log():
    a = Tensor.normal(30, 1, (8, 3, 10, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a.log()
    b_torch = a_torch.log()

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_exp():
    a = Tensor.normal(4, 1, (8, 3, 10, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a.exp()
    b_torch = a_torch.exp()

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_pow():
    a = Tensor.normal(30, 1, (20, 20), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    d = a ** 1.5
    d_torch = a_torch ** 1.5

    d.backward()
    d_torch.sum().backward()

    d.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(d, d_torch)


def test_pow_neg():
    a = Tensor.normal(30, 1, (20, 20), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    c = a ** -1.2
    c_torch = a_torch ** -1.2
    
    c.backward()
    c_torch.sum().backward()

    a.cpu(), c.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(c, c_torch)


def test_div():
    a = Tensor.normal(30, 2, (3, 2, 1), requires_grad=True)
    b = Tensor.normal(30, 2, (3, 2, 3), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    c = a / b
    c_torch = a_torch / b_torch

    c.backward()
    c_torch.sum().backward()

    c.cpu(), b.cpu(), a.cpu()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_div_broadcast():
    a = Tensor.normal(30, 2, (3, 3, 3), requires_grad=True)
    b = Tensor.normal(30, 2, (3, 3, 3), requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu(), b.gpu()

    c = a / 16
    c_torch = a_torch / 16

    d = b * 4
    d_torch = b_torch * 4

    e = c.sum() / d.sum() 
    e_torch = c_torch.sum() / d_torch.sum()

    e.backward()
    e_torch.sum().backward()

    e.cpu(), d.cpu(), c.cpu(), b.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)
    check_val_and_grad(d, d_torch)
    check_val_and_grad(e, e_torch)


def test_relu():
    a = Tensor.normal(0, 2, (8, 3, 10, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.relu()
    b_torch = a_torch.relu()

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sigmoid(): 
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sigmoid()
    b_torch = a_torch.sigmoid()

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_tanh():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.tanh()
    b_torch = a_torch.tanh()

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sum_full_reduce():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum()
    b_torch = a_torch.sum()

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sum_reduce_one_axis():
    a = Tensor.normal(0, 1, (30, 30, 60), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=2)
    b_torch = a_torch.sum(axis=2)

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sum_reduce_axis():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=(1, 2))
    b_torch = a_torch.sum(axis=(1, 2))

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_max_full_reduce():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.max()
    b_torch = a_torch.max()

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_max_reduce_one_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 60), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.max(axis=2)
    b_torch, _ = a_torch.max(axis=2)

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_min_full_reduce():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.min()
    b_torch = a_torch.min()

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_min_reduce_one_axis():
    a = Tensor.normal(0, 1, (30, 30, 60), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.min(axis=2)
    b_torch, _ = a_torch.min(axis=2)

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_reshape():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.reshape((30, 900))
    b_torch = a_torch.reshape((30, 900))

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_transpose():
    a = Tensor.normal(0, 1, (10, 5), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.T()
    b_torch = a_torch.T

    b.backward()
    b_torch.sum().backward()

    a.cpu(), b.cpu()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_slice():
    a = Tensor.normal(0, 1, (30, 40, 20, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a[10:20, :, :12, 5:]
    b_torch = a_torch[10:20, :, :12, 5:]

    b.backward()
    b_torch.sum().backward()

    a.cpu()
    b.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_unsqueeze():
    a = Tensor.normal(0, 1, (3, 3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.unsqueeze(2)
    b_torch = a_torch.unsqueeze(2)

    b_torch.sum().backward()
    b.backward()

    b.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze():
    a = Tensor.normal(0, 1, (30, 1), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.squeeze(1)
    b_torch = a_torch.squeeze(1)

    b_torch.sum().backward()
    b.backward()

    b.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze_no_squeeze():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.squeeze(1)
    b_torch = a_torch.squeeze(1)

    b_torch.sum().backward()
    b.backward()

    b.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze_scalar():
    a = Tensor.normal(0, 1, (1, ), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.squeeze(0)
    b_torch = a_torch.squeeze(0)

    b_torch.sum().backward()
    b.backward()

    b.cpu(), a.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_matmul():
    a = Tensor.normal(0, 1, (30, 15), requires_grad=True)
    b = Tensor.normal(0, 1, (15, 30), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu(), b.gpu()

    c = a @ b
    c_torch = a_torch @ b_torch

    c.backward()
    c_torch.sum().backward()

    c.cpu(), b.cpu(), a.cpu()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_multiple_op():
    a = Tensor.normal(0, 1, (30, 30), requires_grad=True)
    b = Tensor.normal(0, 1, (30, 15), requires_grad=True)
    c = Tensor.normal(30, 3, (30, 15), requires_grad=True)

    a_torch, b_torch, c_torch = create_identical_torch_tensor(a, b, c)

    a.gpu(), b.gpu(), c.gpu()

    d = (a @ b).relu()
    d_torch = (a_torch @ b_torch).relu()

    e = d + c
    e_torch = d_torch + c_torch

    f = e.log()
    f_torch = e_torch.log()

    g = f[:, 3:]
    g_torch = f_torch[:, 3:]

    g.backward()
    g_torch.sum().backward()

    a.cpu(), b.cpu(), c.cpu(), d.cpu(), e.cpu(), f.cpu(), g.cpu()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)
    check_val_and_grad(d, d_torch)
    check_val_and_grad(e, e_torch)
    check_val_and_grad(f, f_torch)
    check_val_and_grad(g, g_torch)