import numpy as np

from nanograd.nn.functional import get_conv1d_output_size, get_conv2d_output_size
import nanograd.nn.module as nnn
from tests.helpers import *

import torch
import torch.nn as nn
from torch.autograd import Variable

from nanograd.tensor import Tensor


SEED = 42


def test_conv1d_output_size():
    input_length = 400
    in_channel, out_channel = 3, 10
    batch_size = 16
    kernel_size = 3
    stride = 2
    padding = 3

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_length))
    inp_torch = create_identical_torch_tensor(inp)

    torch_model = nn.Conv1d(
        in_channels=in_channel, out_channels=out_channel, 
        kernel_size=kernel_size, stride=stride, padding=padding)
    y_torch = torch_model(inp_torch)

    output_shape = get_conv1d_output_size(input_length, kernel_size, stride, padding)

    assert y_torch.shape[2] == output_shape


def test_conv2d_output_size():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 0

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width))
    inp_torch = create_identical_torch_tensor(inp)

    torch_model = nn.Conv2d(
        in_channels=in_channel, out_channels=out_channel, 
        kernel_size=kernel_size, stride=stride, padding=padding)
    y_torch = torch_model(inp_torch)

    output_shape = get_conv2d_output_size(
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    assert y_torch.shape[2] == output_shape[0]
    assert y_torch.shape[3] == output_shape[1]


def test_conv1d_forward():
    input_length = 400
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = 3
    stride = 1
    padding = 2

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_length))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv1d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_conv1d_forward_gpu_no_padding():
    input_length = 400
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = 3
    stride = 1
    padding = 0

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_length))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv1d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_conv1d_forward_gpu():
    input_length = 400
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = 3
    stride = 1
    padding = 3

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_length))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv1d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_conv2d_forward():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 2

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_conv2d_forward_gpu_no_padding():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 0

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_padding_op():
    input_height, input_width = 3, 3
    batch_size = 1
    in_channel, out_channel = 1, 3
    kernel_size = (1, 1)
    stride = 1
    padding = 2

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape
    check_val(y, y_torch)



def test_conv2d_forward_gpu():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 2

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_conv2d_forward_backward_gpu_no_padding():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 0

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width), requires_grad=True)
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    inp.gpu(), model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y_torch.sum().backward()
    y.backward()

    y.cpu(), inp.cpu(), model.cpu()

    assert y.shape == y_torch.shape

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)

"""
def test_conv2d_forward_gpu_padding():

def test_conv2d_forward_backward_gpu():
"""


def test_conv2d_forward_backward():
    input_height, input_width = 200, 200
    batch_size = 16
    in_channel, out_channel = 3, 10
    kernel_size = (3, 3)
    stride = 1
    padding = 2

    inp = Tensor.normal(0, 1, (batch_size, in_channel, input_height, input_width), requires_grad=True)
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y_torch.sum().backward()
    y.backward()

    assert y.shape == y_torch.shape
    check_val_and_grad(y, y_torch)    
    check_val_and_grad(inp, inp_torch)


def test_max_pool_2d_forward():
    inp = Tensor.normal(0, 1, (128, 6, 26, 26))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.MaxPool2d(kernel_size=(2, 2), stride=2))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)
    
    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_max_pool_2d_forward_backward():
    inp = Tensor.normal(0, 1, (128, 6, 26, 26), requires_grad=True)
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.MaxPool2d(kernel_size=(2, 2), stride=2))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()
    
    assert y.shape == y_torch.shape
    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)


def test_avg_pool_2d_forward():
    inp = Tensor.normal(0, 1, (128, 6, 26, 26))
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.AvgPool2d(kernel_size=(2, 2), stride=2))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)
    
    assert y.shape == y_torch.shape
    check_val(y, y_torch)


def test_avg_pool_2d_forward_backward():
    inp = Tensor.normal(0, 1, (128, 6, 26, 26), requires_grad=True)
    inp_torch = create_identical_torch_tensor(inp, dtype=np.double)

    model = nnn.Sequential(nnn.AvgPool2d(kernel_size=(2, 2), stride=2))
    torch_model = get_same_pytorch_mlp(model)

    y = model(inp)
    y_torch = torch_model(inp_torch)

    assert y.shape == y_torch.shape
    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)

