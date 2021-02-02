from nanograd.tensor import Tensor

import torch
import numpy as np

from tests.helpers import check_val_and_grad, create_identical_torch_tensor

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


def test_add_broadcast_2():
    a = Tensor.normal(0, 1, (20, 30), requires_grad=True)
    b = Tensor.normal(0, 1, (30, ), requires_grad=True)

    a_torch, b_torch = create_identical_torch_tensor(a, b)

    c = a + b.T()
    c_torch = a_torch + b_torch.T

    c.backward()
    c_torch.sum().backward()

    check_val_and_grad(c, c_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_add_broadcast_3():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a + 1
    b_torch = a_torch + 1

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_add_broadcast_4():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = 1 + a
    b_torch = 1 + a_torch

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


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


def test_sub_broadcast_2():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a - 1
    b_torch = a_torch - 1

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_sub_broadcast_3():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = 2 - a
    b_torch = 2 - a_torch

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


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


def test_mul_broadcast_2():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a * 2
    b_torch = a_torch * 2

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_mul_broadcast_3():
    a = Tensor.normal(0, 1, (3, 3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = 3 * a
    b_torch = 3 * a_torch

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


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

    b = a.reshape(shape=[400, 1])
    b_torch = a_torch.reshape(shape=[400, 1])

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


def test_pow_exp_neg():
    a = Tensor.normal(30, 1, (20, 20), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a ** (-2)
    b_torch = a_torch ** (-2)

    b_torch.sum().backward()
    b.backward()

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


def test_sigmoid(): 
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.sigmoid()
    b_torch = a_torch.sigmoid()

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_tanh():
    a = Tensor.randn(20, 20, requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.tanh()
    b_torch = a_torch.tanh()

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


def test_mean():
    a = Tensor.normal(30, 1, (5, 5), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.mean(0)
    b_torch = a_torch.mean(0)

    c = a.mean(1)
    c_torch = a_torch.mean(1)

    c_torch.sum().backward()
    b_torch.sum().backward()

    c.backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(c, c_torch)


def test_slice():
    a = Tensor.normal(0, 1, (30, 40, 20, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a[10:20, :, :12, 5:]
    b_torch = a_torch[10:20, :, :12, 5:]

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_min():
    a = Tensor.normal(0, 1, (30, 40, 20, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.min(axis=3).min(axis=0)
    b_torch, _ = a_torch.min(axis=3)
    b_torch, _ = b_torch.min(axis=0)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_max():
    a = Tensor.normal(0, 1, (30, 40, 20, 10), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.max(axis=3).max(axis=1)
    b_torch, _ = a_torch.max(axis=3)
    b_torch, _ = b_torch.max(axis=1)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_unsqueeze():
    a = Tensor.normal(0, 1, (30,), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.unsqueeze(1)
    b_torch = a_torch.unsqueeze(1)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze():
    a = Tensor.normal(0, 1, (30, 1), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.squeeze(1)
    b_torch = a_torch.squeeze(1)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze_no_squeeze():
    a = Tensor.normal(0, 1, (30, 30, 30), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.squeeze(1)
    b_torch = a_torch.squeeze(1)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)


def test_squeeze_scalar():
    a = Tensor.normal(0, 1, (1, ), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = a.squeeze(0)
    b_torch = a_torch.squeeze(0)

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

def test_multiple_2():
    a = Tensor.normal(30, 1, (3, 3), requires_grad=True)
    a_torch = create_identical_torch_tensor(a)

    b = 3 * (1 - a)
    b_torch = 3 * (1 - a_torch)

    b_torch.sum().backward()
    b.backward()

    check_val_and_grad(a, a_torch)
    check_val_and_grad(b, b_torch)

def test_multiple_3():
    a = Tensor.normal(30, 1, (4, 3), requires_grad=True)
    w = Tensor.normal(30, 1, (3,), requires_grad=True)
    b = Tensor.normal(30, 1, (1, 1), requires_grad=True)

    a_torch, w_torch, b_torch = create_identical_torch_tensor(a, w, b)

    o = w.reshape(shape=[1, -1]) * a + b
    o_torch = w_torch * a_torch + b_torch

    o_torch.sum().backward()
    o.backward()

    check_val_and_grad(o, o_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(w, w_torch)
    check_val_and_grad(a, a_torch)
    

def test_vanilla_bn():
    a = Tensor.normal(30, 1, (4, 3, 10), requires_grad=True)
    w = Tensor.normal(30, 1, (10, ), requires_grad=True)
    b = Tensor.normal(30, 1, (10, ), requires_grad=True)

    a_torch, w_torch, b_torch = create_identical_torch_tensor(a, w, b)

    m = a.mean(axis=0)
    m_torch = a_torch.mean(axis=0)

    v = ((a - m) ** 2).mean(axis=0)
    v_torch = ((a_torch - m_torch) ** 2).mean(axis=0)

    a_hat = (a - m) / v.sqrt()
    a_hat_torch = (a_torch - m_torch) / v_torch.sqrt()

    o = w * a_hat + b
    o_torch = w_torch * a_hat_torch + b_torch

    o.backward()
    o_torch.sum().backward()

    check_val_and_grad(o, o_torch)
    check_val_and_grad(w, w_torch)
    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


