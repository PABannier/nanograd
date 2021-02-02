from nanograd.tensor import Tensor, cross_entropy
from nanograd.autograd_engine import *
from nanograd.device import Device
import nanograd.nn.module as nnn

import torch
import numpy as np

from tests.helpers import *


def test_cross_entropy():
    num_classes = 10
    batch_size = 64

    predicted = Tensor.normal(5, 2, (batch_size, num_classes), requires_grad=True)
    target = Tensor.randint(0, 9, (batch_size, ))
    pred_torch, target_torch = create_identical_torch_tensor(predicted, target)

    target_torch = target_torch.type(torch.LongTensor)

    loss = cross_entropy(predicted, target)
    loss_torch = torch.nn.functional.cross_entropy(pred_torch, target_torch)

    loss_torch.sum().backward()
    loss.backward()

    check_val_and_grad(loss, loss_torch)
    check_val_and_grad(predicted, pred_torch)
    check_val_and_grad(target, target_torch)

def test_mse_loss():
    batch_size = 64

    predicted = Tensor.normal(5, 2, (batch_size, 1), requires_grad=True)
    target = Tensor.normal(5, 3, (batch_size, 1))
    pred_torch, target_torch = create_identical_torch_tensor(predicted, target)

    loss = nnn.MSELoss()(predicted, target)
    loss_torch = torch.nn.functional.mse_loss(pred_torch, target_torch)

    loss_torch.sum().backward()
    loss.backward()

    check_val_and_grad(loss, loss_torch)
    check_val_and_grad(predicted, pred_torch)
    check_val_and_grad(target, target_torch)

def test_cross_entropy_gpu():
    num_classes = 10
    batch_size = 64

    predicted = Tensor.normal(5, 2, (batch_size, num_classes), requires_grad=True)
    target = Tensor.randint(0, 9, (batch_size,))
    pred_torch, target_torch = create_identical_torch_tensor(predicted, target)

    target_torch = target_torch.type(torch.LongTensor)

    predicted.gpu(), target.gpu()

    loss = cross_entropy(predicted, target)
    loss_torch = torch.nn.functional.cross_entropy(pred_torch, target_torch)

    loss_torch.sum().backward()
    loss.backward()

    loss = loss.cpu()
    predicted = predicted.cpu()
    target = target.cpu()

    check_val_and_grad(loss, loss_torch)
    check_val_and_grad(predicted, pred_torch)
    check_val_and_grad(target, target_torch)

def test_mse_loss_gpu():
    batch_size = 64

    predicted = Tensor.normal(5, 2, (batch_size, 1), requires_grad=True)
    target = Tensor.normal(5, 3, (batch_size, 1))
    pred_torch, target_torch = create_identical_torch_tensor(predicted, target)

    predicted.gpu(), target.gpu()

    loss = nnn.MSELoss()(predicted, target)
    loss_torch = torch.nn.functional.mse_loss(pred_torch, target_torch)

    loss_torch.sum().backward()
    loss.backward()

    loss = loss.cpu()
    predicted = predicted.cpu()
    target = target.cpu()

    check_val_and_grad(loss, loss_torch)
    check_val_and_grad(predicted, pred_torch)
    check_val_and_grad(target, target_torch)