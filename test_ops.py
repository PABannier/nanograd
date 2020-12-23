from tensor import Tensor
from autograd_engine import *
from nn.functional import *

import torch
import numpy as np

from utils import check_val_and_grad, create_identical_torch_tensor

def test_add():
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)
    c = Tensor(3, requires_grad=True)

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


def test_add_broadcast():
    a = Tensor.randn(3, 3, requires_grad=True)
    b = Tensor.randn(3, 1, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a + b
    c_torch = a_torch + b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_sub_broadcast():
    a = Tensor.randn(3, 2, 1, requires_grad=True)
    b = Tensor.randn(3, 1, 1, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a - b
    c_torch = a_torch - b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_neg():
    a = Tensor.randn(30, 40, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = -a
    b_torch = -a_torch

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_mul():
    a = Tensor.randn(30, 40, requires_grad=True)
    b = Tensor.randn(30, 40, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a * b
    c_torch = a_torch * b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_mul_broadcast(): 
    a = Tensor.randn(3, 1, 3, requires_grad=True)
    b = Tensor.randn(3, 1, 1, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a * b
    c_torch = a_torch * b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_div():
    a = Tensor.randn(30, 40, requires_grad=True)
    b = Tensor.randn(30, 40, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a / b
    c_torch = a_torch / b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_div_broadcast():
    a = Tensor.randn(3, 2, 3, requires_grad=True)
    b = Tensor.randn(3, 1, 1, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a / b
    c_torch = a_torch / b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_reshape():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.reshape((400, 1))
    b_torch = a_torch.reshape((400, 1))

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_transpose():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.T()
    b_torch = a_torch.T

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_sum():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.sum()
    b_torch = a_torch.sum()

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_matmul():
    a = Tensor.randn(3, 4, requires_grad=True)
    b = Tensor.randn(4, 2, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a @ b
    c_torch = a_torch @ b_torch

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)

def test_pow():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a ** 3
    b_torch = a_torch ** 3

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_relu(): 
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.relu()
    b_torch = a_torch.relu()

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_log():
    a = Tensor.normal(30, 1, (5, 5), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.log()
    b_torch = a_torch.log()

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_exp():
    a = Tensor.normal(30, 1, (5, 5), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.log()
    b_torch = a_torch.log()

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_sqrt():
    a = Tensor.normal(30, 1, (5, 5), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)
    
    b = a.sqrt()
    b_torch = a_torch.sqrt()

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_multiple():
    a = Tensor.randn(2, 3, requires_grad=True)
    b = Tensor.randn(2, 3, requires_grad=True)
    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a / b
    c_torch = a_torch / b_torch

    d = a - b
    d_torch = a_torch - b_torch

    e = c + d
    e_torch = c_torch + d_torch

    e.backward()
    e_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)
    check_val_and_grad(d, d_torch)
    check_val_and_grad(e, e_torch)