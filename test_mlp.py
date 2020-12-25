import numpy as np
from utils import *

import torch
import torch.nn

from tensor import Tensor
from nn.module import BatchNorm1d, Linear, ReLU, Sequential
from nn.loss import CrossEntropyLoss
from optim.optimizer import SGD

SEED = 42


# ****** Layer tests ******


def test_linear_forward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20))
    _test_forward(model)
    check_model_param_settings(model)


def test_linear_backward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20))
    _test_forward_backward(model)


def test_relu_forward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU())
    _test_forward(model)
    check_model_param_settings(model)


def test_relu_backward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU())
    _test_forward_backward(model)


def test_multiple_layer_forward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU(), Linear(20, 40), ReLU())
    _test_forward(model)
    check_model_param_settings(model)


def test_multiple_layer_backward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU(), Linear(20, 40), ReLU())
    _test_forward_backward(model)


def test_batchnorm_forward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    _test_forward(model)
    check_model_param_settings(model)


def test_batchnorm_backward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    _test_forward_backward(model)


# ****** SGD tests ******


def test_linear_relu_step():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU())
    optimizer = SGD(model.parameters())
    _test_step(model, optimizer, 5, 5)


def test_multiple_layer_relu_step():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20),  ReLU(), Linear(20, 30), ReLU())
    optimizer = SGD(model.parameters())
    _test_step(model, optimizer, 5, 5)


def test_linear_batchnorm_relu_train_eval():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    optimizer = SGD(model.parameters())
    _test_step(model, optimizer, 5, 5)


def test_big_linear_batchnorm_relu_train_eval():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU())
    optimizer = SGD(model.parameters())
    _test_step(model, optimizer, 5, 5)


# ****** Cross-entropy tests ******


def test_linear_xeloss_forward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20))
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    _test_forward(model, criterion=criterion)


def test_linear_xeloss_backward():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20))
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    _test_forward_backward(model, criterion=criterion)


def test_big_linear_bn_relu_xeloss_train_eval():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU(), Linear(20, 30), BatchNorm1d(30), ReLU())
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    _test_step(model, optimizer, 5, 5, criterion=criterion)


def test_big_linear_relu_xeloss_train_eval():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU(), Linear(20, 30), ReLU())
    optimizer = SGD(model.parameters())
    criterion = CrossEntropyLoss()
    _test_step(model, optimizer, 5, 5, criterion=criterion)


# ****** Momentum tests ******


def test_linear_momentum():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU())
    optimizer = SGD(model.parameters(), momentum=0.9)
    _test_step(model, optimizer, 5, 0)


def test_big_linear_batchnorm_relu_xeloss_momentum():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), BatchNorm1d(20), ReLU(),
                       Linear(20, 30), BatchNorm1d(30), ReLU())
    optimizer = SGD(model.parameters(), momentum=0.9)
    criterion = CrossEntropyLoss()
    _test_step(model, optimizer, 5, 5, criterion=criterion)


def test_big_linear_relu_xeloss_momentum():
    np.random.seed(SEED)
    model = Sequential(Linear(10, 20), ReLU(),
                       Linear(20, 30), ReLU())
    optimizer = SGD(model.parameters(), momentum = 0.9)
    criterion = CrossEntropyLoss()
    _test_step(model, optimizer, 5, 5, criterion=criterion)


# ****** Helpers ******


def _test_forward(model, criterion=None, batch_size=(2,5)):
    """
        Tests forward, printing whether a mismatch occurs in forward.
    """
    pytorch_model = get_same_pytorch_mlp(model)
    batch_size = np.random.randint(*batch_size) if type(batch_size) == tuple\
        else batch_size
    x, y = generate_dataset_for_mytorch_model(model, batch_size)
    pytorch_criterion = get_same_pytorch_criterion(criterion)

    forward_(model, criterion, pytorch_model, pytorch_criterion, x, y)


def _test_forward_backward(model, criterion=None, batch_size=(2,5)):
    """
        Tests forward and back, printing whether a mismatch occurs in forward or
        backwards.
    """
    pytorch_model = get_same_pytorch_mlp(model)
    batch_size = np.random.randint(*batch_size) if type(batch_size) == tuple\
        else batch_size
    x, y = generate_dataset_for_mytorch_model(model, batch_size)
    pytorch_criterion = get_same_pytorch_criterion(criterion)

    (mx, my, px, py) = forward_(model, criterion, pytorch_model,
                                pytorch_criterion, x, y)

    backward_(mx, my, model, px, py, pytorch_model)


def _test_step(model, optimizer, train_steps, eval_steps,
               criterion=None, batch_size=(2, 5)):
    """
        Tests subsequent forward, back, and update operations, printing whether
        a mismatch occurs in forward or backwards.
    """
    pytorch_model = get_same_pytorch_mlp(model)
    pytorch_optimizer = get_same_pytorch_optimizer(optimizer, pytorch_model)
    pytorch_criterion = get_same_pytorch_criterion(criterion)
    batch_size = np.random.randint(*batch_size) if type(batch_size) == tuple\
        else batch_size
    x, y = generate_dataset_for_mytorch_model(model, batch_size)

    model.train()
    pytorch_model.train()
    for s in range(train_steps):
        pytorch_optimizer.zero_grad()
        optimizer.zero_grad()

        (mx, my, px, py) = forward_(model, criterion,
                                    pytorch_model, pytorch_criterion, x, y)

        backward_(mx, my, model, px, py, pytorch_model)

        pytorch_optimizer.step()
        optimizer.step()
        check_model_param_settings(model)

    model.eval()
    pytorch_model.eval()
    for s in range(eval_steps):
        pytorch_optimizer.zero_grad()
        optimizer.zero_grad()

        (mx, my, px, py) = forward_(model, criterion,
                                    pytorch_model, pytorch_criterion, x, y)

    check_model_param_settings(model)