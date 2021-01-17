from tensor import Tensor, Device
from autograd_engine import *
from nn.functional import *

import torch
import numpy as np

from tests.helpers import *

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

def test_sum_full_reduce_forward():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum()
    b_torch = a_torch.sum()

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch, atol=1e-4)
    check_val_and_grad(a, a_torch, atol=1e-4)

def test_sum_reduce_one_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 60), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=2)
    b_torch = a_torch.sum(axis=2)

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch, atol=1e-4)
    check_val_and_grad(a, a_torch, atol=1e-4)

def test_sum_reduce_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=(1, 2))
    b_torch = a_torch.sum(axis=(1, 2))

    b.backward()
    b_torch.sum().backward()

    b.cpu(), a.cpu()

    check_val_and_grad(b, b_torch, atol=1e-4)
    check_val_and_grad(a, a_torch, atol=1e-4)

def test_max_full_reduce_forward():
    a = Tensor.normal(0, 1, (30, 30, 30))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.max()
    b_torch = a_torch.max()

    b.cpu()

    check_val(b, b_torch)

def test_max_reduce_one_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 60))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.max(axis=2)
    b_torch, _ = a_torch.max(axis=2)

    b.cpu()

    check_val(b, b_torch)

def test_min_full_reduce_forward():
    a = Tensor.normal(0, 1, (30, 30, 30))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.min()
    b_torch = a_torch.min()

    b.cpu()

    check_val(b, b_torch)

def test_min_reduce_one_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 60))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.min(axis=2)
    b_torch, _ = a_torch.min(axis=2)

    b.cpu()

    check_val(b, b_torch)

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