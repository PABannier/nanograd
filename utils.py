import numpy as np
import torch

def check_val(nano_tensor, torch_tensor):
    assert np.allclose(nano_tensor.data, torch_tensor.data.numpy())

def check_grad(nano_tensor, torch_tensor):
    if nano_tensor.grad is not None and torch_tensor.grad is not None:
        assert np.allclose(nano_tensor.grad.data, torch_tensor.grad.numpy(), atol=1e-3)
    elif nano_tensor.grad is not None and torch_tensor.grad is None:
        raise Exception("NanoTensor is not None, while torchtensor is None")
    elif nano_tensor.grad is None and torch_tensor.grad is not None:
        raise Exception("NanoTensor is None, while torchtensor is not None")
    else:
        pass

def check_val_and_grad(nano_tensor, torch_tensor):
    check_val(nano_tensor, torch_tensor)
    check_grad(nano_tensor, torch_tensor)

def create_identical_torch_tensor(*args):
    torch_tensors = []
    for arg in args:
        t = torch.tensor(arg.data.astype(np.float32), requires_grad=arg.requires_grad)
        torch_tensors.append(t)
    return tuple(torch_tensors) if len(torch_tensors) > 1 else torch_tensors[0]