from tests.helpers import check_val_and_grad, create_identical_torch_tensor
from tests.helpers import (get_same_pytorch_model, check_model_parameters, 
                          get_same_pytorch_optimizer)

from nanograd.tensor import Tensor
import nanograd.nn.module as nnn 
import nanograd.optim.optimizer as optim

import torch.nn as nn

import numpy as np


def test_simple_linear_model_sgd():
    inp, targ = Tensor.normal(0, 1, (8, 256)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Linear(256, 128), nnn.LeakyReLU(), 
                           nnn.Linear(128, 1), nnn.ReLU())
    optimizer = optim.SGD(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch)

def test_linear_bn1_model_sgd():
    inp, targ = Tensor.normal(0, 1, (8, 256)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Linear(256, 128), nnn.BatchNorm1d(128), 
                           nnn.Linear(128, 64), nnn.BatchNorm1d(64),
                           nnn.LeakyReLU(), nnn.Linear(64, 1))
    optimizer = optim.SGD(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch)

def test_conv1d_mpool1d_model_sgd():
    inp, targ = Tensor.normal(0, 1, (8, 4, 100)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv1d(4, 32, 3, 3, 2), nnn.MaxPool1d(2, 2), nnn.ReLU(),
                           nnn.Conv1d(32, 64, 2, 2, 2), nnn.AvgPool1d(2, 2), nnn.Flatten(),
                           nnn.Linear(320, 1))
    optimizer = optim.SGD(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch, atol=1e-4)

def test_conv2d_model_sgd():
    inp, targ = Tensor.normal(0, 1, (8, 3, 30, 30)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv2d(3, 32, 2, 2), nnn.MaxPool2d(2), nnn.ReLU(),
                           nnn.Conv2d(32, 64, 3, 3), nnn.AvgPool2d(2), nnn.ReLU(),
                           nnn.Flatten(), nnn.Linear(64, 1))
    optimizer = optim.SGD(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()
    
    check_model_parameters(model, model_torch, atol=1e-4)

def test_conv2d_bn2d_model_sgd():
    inp, targ = Tensor.normal(0, 1, (8, 3, 30, 30)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv2d(3, 32, 2, 2), nnn.BatchNorm2d(32), nnn.ReLU(),
                           nnn.Conv2d(32, 64, 3, 3), nnn.BatchNorm2d(64), nnn.ReLU(),
                           nnn.Flatten(), nnn.Linear(1600, 1))
    optimizer = optim.SGD(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()
    
    check_model_parameters(model, model_torch, atol=1e-4)


def test_simple_linear_model_adam():
    inp, targ = Tensor.normal(0, 1, (8, 256)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Linear(256, 128), nnn.LeakyReLU(), 
                           nnn.Linear(128, 1), nnn.ReLU())
    optimizer = optim.Adam(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch)

def test_linear_bn1_model_adam():
    inp, targ = Tensor.normal(0, 1, (8, 256)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Linear(256, 128), nnn.BatchNorm1d(128), 
                           nnn.Linear(128, 64), nnn.BatchNorm1d(64),
                           nnn.LeakyReLU(), nnn.Linear(64, 1))
    optimizer = optim.Adam(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch)

def test_conv1d_mpool1d_model_adam():
    inp, targ = Tensor.normal(0, 1, (8, 4, 100)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv1d(4, 32, 3, 3, 2), nnn.MaxPool1d(2, 2), nnn.ReLU(),
                           nnn.Conv1d(32, 64, 2, 2, 2), nnn.AvgPool1d(2, 2), nnn.Flatten(),
                           nnn.Linear(320, 1))
    optimizer = optim.Adam(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()

    check_model_parameters(model, model_torch)

def test_conv2d_model_adam():
    inp, targ = Tensor.normal(0, 1, (8, 3, 30, 30)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv2d(3, 32, 2, 2), nnn.MaxPool2d(2), nnn.ReLU(),
                           nnn.Conv2d(32, 64, 3, 3), nnn.AvgPool2d(2), nnn.ReLU(),
                           nnn.Flatten(), nnn.Linear(64, 1))
    optimizer = optim.Adam(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()
    
    check_model_parameters(model, model_torch, atol=6e-5)

def test_conv2d_bn2d_model_adam():
    inp, targ = Tensor.normal(0, 1, (8, 3, 30, 30)), Tensor.normal(0, 1, (8, 1))
    model = nnn.Sequential(nnn.Conv2d(3, 32, 2, 2), nnn.BatchNorm2d(32), nnn.ReLU(),
                           nnn.Conv2d(32, 64, 3, 3), nnn.BatchNorm2d(64), nnn.ReLU(),
                           nnn.Flatten(), nnn.Linear(1600, 1))
    optimizer = optim.Adam(model.parameters())

    inp_torch, targ_torch = create_identical_torch_tensor(inp, targ)
    model_torch = get_same_pytorch_model(model)
    optimizer_torch = get_same_pytorch_optimizer(optimizer, model_torch)

    for _ in range(5):
        y = model(inp)
        y_torch = model_torch(inp_torch)

        loss = nnn.MSELoss()(y, targ)
        loss_torch = nn.MSELoss()(y_torch, targ_torch)

        loss.backward()
        loss_torch.backward()

        optimizer.step()
        optimizer_torch.step()
    
    check_model_parameters(model, model_torch, atol=1e-4)
