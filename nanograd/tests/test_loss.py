from tensor import Tensor
from autograd_engine import *
from nn.functional import *

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

