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

    a.gpu()
    b.gpu()

    c = a + b
    c_torch = a_torch + b_torch

    c.cpu()

    check_val(c, c_torch)

def test_neg_forward():
    a = Tensor.normal(0, 1, (8, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = -a
    b_torch = -a_torch
    b.cpu()

    check_val(b, b_torch)

def test_sub_forward():
    a = Tensor.normal(0, 1, (3, 3, 3))
    b = Tensor.normal(0, 1, (3, 3, 1))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    d = a - b
    d_torch = a_torch - b_torch

    d.cpu()

    check_val(d, d_torch)

def test_mul_forward():
    a = Tensor.normal(30, 2, (3, 2, 1))
    b = Tensor.normal(30, 2, (3, 2, 3))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    c = a * b
    c_torch = a_torch * b_torch

    c.cpu()

    check_val(c, c_torch)

def test_log_forward():
    a = Tensor.normal(30, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a.log()
    b_torch = a_torch.log()
    b.cpu()

    check_val(b, b_torch)

def test_exp_forward():
    a = Tensor.normal(4, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a.exp()
    b_torch = a_torch.exp()
    b.cpu()

    check_val(b, b_torch)

def test_pow_forward():
    a = Tensor.normal(4, 1, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a ** 3.0
    b_torch = a_torch ** 3.0

    c = a ** -1.0
    c_torch = a_torch ** -1.0

    d = a ** 1.5
    d_torch = a_torch ** 1.5

    b.cpu()
    c.cpu()
    d.cpu()

    check_val(b, b_torch)
    check_val(c, c_torch)
    check_val(d, d_torch)

def test_div_forward():
    a = Tensor.normal(30, 2, (3, 2, 1))
    b = Tensor.normal(30, 2, (3, 2, 3))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu()
    b.gpu()

    c = a / b
    c_torch = a_torch / b_torch

    c.cpu()

    check_val(c, c_torch, atol=1e-3)

def test_relu_forward():
    a = Tensor.normal(0, 2, (8, 3, 10, 10))
    a_torch = create_identical_torch_tensor(a)
    a.gpu()

    b = a.relu()
    b_torch = a_torch.relu()

    b.cpu()

    check_val(b, b_torch)

def test_sigmoid_forward(): 
    a = Tensor.randn(20, 20)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sigmoid()
    b_torch = a_torch.sigmoid()
    b.cpu()

    check_val(b, b_torch)

def test_tanh_forward():
    a = Tensor.randn(20, 20)
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.tanh()
    b_torch = a_torch.tanh()
    b.cpu()

    check_val(b, b_torch)

def test_sum_full_reduce_forward():
    a = Tensor.normal(0, 1, (30, 30, 30))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum()
    b_torch = a_torch.sum()

    b.cpu()

    check_val(b, b_torch, atol=1e-4)

def test_sum_reduce_one_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 60))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=2)
    b_torch = a_torch.sum(axis=2)

    b.cpu()

    check_val(b, b_torch, atol=1e-4)

def test_sum_reduce_axis_forward():
    a = Tensor.normal(0, 1, (30, 30, 30))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.sum(axis=(1, 2))
    b_torch = a_torch.sum(axis=(1, 2))

    b.cpu()

    check_val(b, b_torch, atol=1e-4)

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

def test_reshape_forward():
    a = Tensor.normal(0, 1, (30, 30, 30))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.reshape((30, 900))
    b_torch = a_torch.reshape((30, 900))

    b.cpu()

    check_val(b, b_torch)

def test_transpose_forward():
    a = Tensor.normal(0, 1, (10, 5))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a.T()
    b_torch = a_torch.T

    b.cpu()

    check_val(b, b_torch)

def test_slice():
    a = Tensor.normal(0, 1, (30, 40, 20, 10))
    a_torch = create_identical_torch_tensor(a)

    a.gpu()

    b = a[10:20, :, :12, 5:]
    b_torch = a_torch[10:20, :, :12, 5:]

    a.cpu()
    b.cpu()

    check_val(a, a_torch)
    check_val(b, b_torch)

def test_matmul():
    a = Tensor.normal(0, 1, (30, 15))
    b = Tensor.normal(0, 1, (15, 30))

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    a.gpu(), b.gpu()

    c = a @ b
    c_torch = a_torch @ b_torch

    c.cpu()

    check_val(c, c_torch)