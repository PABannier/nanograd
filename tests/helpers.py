import numpy as np

import torch
from nanograd.tensor import Tensor

import nanograd.nn.module as nnn 
import nanograd.optim.optimizer as optim

import torch.nn as nn
import torch.optim


def check_val(nano_tensor, torch_tensor, atol=1e-5):
    np.testing.assert_allclose(nano_tensor.data, torch_tensor.data.numpy(), atol=atol)

def check_grad(nano_tensor, torch_tensor, atol=1e-5):
    if nano_tensor.grad is not None and torch_tensor.grad is not None:
        np.testing.assert_allclose(nano_tensor.grad.data, torch_tensor.grad.numpy(), atol=atol)

def check_val_and_grad(nano_tensor, torch_tensor, atol=1e-5, atol_grad=1e-5):
    assert type(nano_tensor).__name__ == "Tensor", f"Expected Tensor object, got {type(nano_tensor).__name__}"
    check_val(nano_tensor, torch_tensor, atol=atol)
    check_grad(nano_tensor, torch_tensor, atol=atol_grad)

def create_identical_torch_tensor(*args):
    torch_tensors = []
    for arg in args:
        t = torch.tensor(arg.data, requires_grad=arg.requires_grad, dtype=torch.float32)
        torch_tensors.append(t)
    return tuple(torch_tensors) if len(torch_tensors) > 1 else torch_tensors[0]
    
def get_same_pytorch_model(model):
    layers = []
    
    for l in model.layers:
        if isinstance(l, nnn.Linear):
            in_f, out_f = l.in_features, l.out_features
            weight = torch.Tensor(l.weight.data)
            bias = torch.Tensor(l.bias.data)
            torch_layer = nn.Linear(in_f, out_f)
            torch_layer.weight, torch_layer.bias = nn.Parameter(weight), nn.Parameter(bias)
            layers.append(torch_layer)
        
        elif isinstance(l, (nnn.BatchNorm1d, nnn.BatchNorm2d)):
            mom, eps = l.momentum, l.eps
            torch_layer = nn.BatchNorm1d(l.num_features, momentum=mom, eps=eps) if isinstance(l, nnn.BatchNorm1d) else nn.BatchNorm2d(l.num_features, momentum=mom, eps=eps)
            weight, bias = torch.Tensor(l.weight.data), torch.Tensor(l.bias.data)
            run_mean, run_var = torch.Tensor(l.running_mean.data), torch.Tensor(l.running_var.data)
            torch_layer.weight, torch_layer.bias = nn.Parameter(weight), nn.Parameter(bias)
            torch_layer.running_mean, torch_layer.running_var = nn.Parameter(run_mean, requires_grad=False), nn.Parameter(run_var, requires_grad=False)
            layers.append(torch_layer)

        elif isinstance(l, (nnn.Conv1d, nnn.Conv2d)):
            in_channel, out_channel = l.in_channel, l.out_channel
            stride, padding = l.stride, l.padding[0]
            print(padding)
            kernel_size = l.kernel_size
            weight, bias = torch.Tensor(l.weight.data), torch.Tensor(l.bias.data)
            torch_layer = (nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding) if isinstance(l, nnn.Conv1d) 
                          else nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding))
            torch_layer.weight, torch_layer.bias = nn.Parameter(weight), nn.Parameter(bias)
            layers.append(torch_layer)
        
        elif isinstance(l, (nnn.MaxPool1d, nnn.AvgPool1d)):
            size, stride = l.kernel_size, l.stride
            torch_layer = nn.MaxPool1d(size, stride) if isinstance(l, nnn.MaxPool1d) else nn.AvgPool1d(size, stride)
            layers.append(torch_layer)

        elif isinstance(l, (nnn.MaxPool2d, nnn.AvgPool2d)):
            size, stride = l.kernel_size, l.stride
            torch_layer = nn.MaxPool2d(size, stride) if isinstance(l, nnn.MaxPool2d) else nn.AvgPool2d(size, stride)
            layers.append(torch_layer)
        
        elif isinstance(l, nnn.Flatten):
            layers.append(nn.Flatten())
        
        elif isinstance(l, nnn.ReLU):
            layers.append(nn.ReLU())
        
        elif isinstance(l, nnn.Sigmoid):
            layers.append(nn.Sigmoid())
        
        elif isinstance(l, nnn.Tanh):
            layers.append(nn.Tanh())
        
        elif isinstance(l, nnn.LeakyReLU):
            layers.append(nn.LeakyReLU(float(l.alpha.data)))
        
        else:
            raise NotImplementedError('Not supported Nanograd layer')
    
    return nn.Sequential(*layers)

def get_same_pytorch_optimizer(optimizer, torch_model):
    if isinstance(optimizer, optim.SGD):
        lr, mom = float(optimizer.lr), float(optimizer.momentum)
        return torch.optim.SGD(torch_model.parameters(), lr, mom)

    elif isinstance(optimizer, optim.Adam):
        lr, beta1 = float(optimizer.lr), float(optimizer.beta1)
        beta2, eps = float(optimizer.beta2), float(optimizer.eps)
        return torch.optim.Adam(torch_model.parameters(), lr, (beta1, beta2), eps)
    
    elif isinstance(optimizer, optim.AdamW):
        lr, reg = float(optimizer.lr), float(optimizer.reg)
        beta1, beta2 = float(optimizer.beta1), float(optimizer.beta2)
        eps = float(optimizer.eps)
        return torch.optim.AdamW(torch_model.parameters(), lr, (beta1, beta2), eps, reg)
    
    else:
        raise NotImplementedError('Not supported Nanograd optimizer')

def check_model_parameters(ng_model, pytorch_model, atol=1e-5):
    pytorch_layers = [module for module in pytorch_model.modules() if type(module) != nn.Sequential]
    for ng_l, pt_l in zip(ng_model.layers, pytorch_layers):
        if isinstance(ng_l, nnn.Linear):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol)

            if ng_l.weight.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol)
        
        elif isinstance(ng_l, (nnn.BatchNorm1d, nnn.BatchNorm2d)):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.running_mean.data, pt_l.running_mean.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.running_var.data, pt_l.running_var.detach().numpy(), atol=atol)

            if ng_l.weight.grad and ng_l.bias.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol)
        
        elif isinstance(ng_l, (nnn.Conv1d, nnn.Conv2d)):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol)

            if ng_l.weight.grad and ng_l.bias.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol)
