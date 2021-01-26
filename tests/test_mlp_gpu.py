import numpy as np

from tests.helpers import *

import torch
import torch.nn as nn
from torch.autograd import Variable

from nanograd.tensor import Tensor

SEED = 42


# ****** Layer tests ******


def test_linear_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20))
    forward_test(model, test_on_gpu=True)
    check_model_param_settings(model)

def test_linear_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20))
    forward_backward_test(model, test_on_gpu=True)

def test_relu_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU())
    forward_test(model, test_on_gpu=True)
    check_model_param_settings(model)

def test_relu_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU())
    forward_backward_test(model, test_on_gpu=True)

def test_multiple_layer_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(
        nnn.Linear(10, 20), nnn.ReLU(), 
        nnn.Linear(20, 40), nnn.ReLU()
    )
    forward_test(model, test_on_gpu=True)
    check_model_param_settings(model)

def test_multiple_layer_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(
        nnn.Linear(10, 20), nnn.ReLU(), 
        nnn.Linear(20, 40), nnn.ReLU())
    forward_backward_test(model, test_on_gpu=True)

def test_flatten_forward():
    np.random.seed(SEED)
    inp = Tensor.normal(0, 1, (8, 30, 3), requires_grad=True)
    inp_torch = create_identical_torch_tensor(inp)

    model = nnn.Flatten()
    torch_model = nn.Flatten()

    inp.gpu()
    model.gpu()

    y = model(inp)
    y_torch = torch_model(inp_torch)

    y_torch.sum().backward()
    y.backward()

    y.cpu(), inp.cpu()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)

def test_batchnorm1d_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU())
    forward_test(model, test_on_gpu=True)
    check_model_param_settings(model)


def test_batchnorm1d_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU())
    forward_backward_test(model, test_on_gpu=True)


def test_batchnorm2d_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 1), nnn.BatchNorm2d(10), nnn.ReLU())
    pytorch_model = get_same_pytorch_mlp(model)

    shape = (16, 3, 10, 10)
    x = Tensor.normal(0, 1, shape)
    x_torch = create_identical_torch_tensor(x).double()

    x.gpu()
    model.gpu()

    y = model(x)
    y_torch = pytorch_model(x_torch)

    x.cpu(), y.cpu()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(x, x_torch)

"""
def test_batchnorm2d_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 1), nnn.BatchNorm2d(10), nnn.ReLU())
    pytorch_model = get_same_pytorch_mlp(model)

    shape = (16, 3, 10, 10)
    x = Tensor.normal(0, 1, shape)
    x_torch = create_identical_torch_tensor(x).double()

    x.gpu()
    model.gpu()

    y = model(x)
    y_torch = pytorch_model(x_torch)

    y.backward()
    y_torch.sum().backward()

    x.cpu(), y.cpu()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(x, x_torch)
"""


# ****** SGD tests ******


def test_linear_relu_step():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU())
    optimizer = SGD(model.parameters())
    step_test(model, optimizer, 5, 5, test_on_gpu=True)


def test_multiple_layer_relu_step():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU(), 
                           nnn.Linear(20, 30), nnn.ReLU())
    optimizer = SGD(model.parameters())
    step_test(model, optimizer, 5, 5, test_on_gpu=True)

"""
def test_linear_batchnorm_relu_train_eval():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU())
    optimizer = SGD(model.parameters())
    step_test(model, optimizer, 5, 5, test_on_gpu=True)
"""

"""
def test_big_linear_batchnorm_relu_train_eval():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU())
    optimizer = SGD(model.parameters())
    step_test(model, optimizer, 5, 5, test_on_gpu=True)
"""


# ****** Cross-entropy tests ******


def test_linear_xeloss_forward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20))
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    forward_test(model, criterion=criterion, test_on_gpu=True)


def test_linear_xeloss_backward():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20))
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    forward_backward_test(model, criterion=criterion, test_on_gpu=True)

"""
def test_big_linear_bn_relu_xeloss_train_eval():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU(), 
                           nnn.Linear(20, 30), nnn.BatchNorm1d(30), nnn.ReLU())
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    step_test(model, optimizer, 5, 5, criterion=criterion, test_on_gpu=True)
"""


def test_big_linear_relu_xeloss_train_eval():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU(), 
                           nnn.Linear(20, 30), nnn.ReLU())
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    step_test(model, optimizer, 5, 5, criterion=criterion, test_on_gpu=True)



# ****** Momentum tests ******


def test_linear_momentum():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU())
    optimizer = SGD(model.parameters(), momentum=0.9)
    step_test(model, optimizer, 5, 0, test_on_gpu=True)

"""
def test_big_linear_batchnorm_relu_xeloss_momentum():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.BatchNorm1d(20), nnn.ReLU(),
                           nnn.Linear(20, 30), nnn.BatchNorm1d(30), nnn.ReLU())
    optimizer = SGD(model.parameters(), momentum=0.9)
    criterion = CrossEntropyLoss()
    step_test(model, optimizer, 5, 5, criterion=criterion, test_on_gpu=True)
"""


def test_big_linear_relu_xeloss_momentum():
    np.random.seed(SEED)
    model = nnn.Sequential(nnn.Linear(10, 20), nnn.ReLU(),
                           nnn.Linear(20, 30), nnn.ReLU())
    optimizer = SGD(model.parameters(), momentum = 0.9)
    criterion = CrossEntropyLoss()
    step_test(model, optimizer, 5, 5, criterion=criterion, test_on_gpu=True)