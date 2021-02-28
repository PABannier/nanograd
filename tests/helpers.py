import numpy as np

import torch
from nanograd.tensor import Tensor
from nanograd.device import Device

import nanograd.nn.module as nnn 
import nanograd.optim.optimizer as optim

import torch.nn as nn
import torch.optim

from tqdm import trange
import time
import traceback


#######################
### CUSTOM PROFILER ###
#######################

class OpProfiler:
    def __init__(self, func_name=None, device=None):
        self.nano_fp, self.nano_bp = 0, 0
        self.torch_fp, self.torch_bp = 0, 0

        self.func_name = func_name
        self.device = device

        self.op_id = ""
        self.start = 0
    
    def __call__(self, op_id):
        assert op_id in [k for k in self.__dict__.keys() \
                         if not k.startswith('__') and not callable(k)], 'Wrong operation id'
        self.op_id = op_id
        return self

    def __enter__(self):
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            return False
        
        op_time = (time.time() - self.start) * 1000
        setattr(self, self.op_id, op_time)
        return True

    def print(self):
        print(f'{str(self.func_name)},  device: {str(self.device.name).lower()},  torch/nanograd ' + 
              f'fp: {self.torch_fp:.2f} / {self.nano_fp:.2f} ms, ' + 
              f'bp: {self.torch_bp:.2f} / {self.nano_bp:.2f} ms')


#######################
## TENSOR COMPARISON ##
#######################

def check_val(nano_tensor, torch_tensor, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(nano_tensor.data, torch_tensor.data.numpy(), atol=atol, rtol=rtol)

def check_grad(nano_tensor, torch_tensor, atol=1e-6, rtol=1e-3):
    if nano_tensor.grad is not None and torch_tensor.grad is not None:
        np.testing.assert_allclose(nano_tensor.grad.data, torch_tensor.grad.numpy(), atol=atol, rtol=rtol)
    elif nano_tensor.grad is None and torch_tensor is not None:
        raise Exception('Nanograd tensor is None, while PyTorch tensor is not None')

def check_val_and_grad(nano_tensor, torch_tensor, atol=1e-6, rtol=1e-3, atol_grad=1e-6, rtol_grad=1e-3):
    assert type(nano_tensor).__name__ == "Tensor", f"Expected Tensor object, got {type(nano_tensor).__name__}"
    check_val(nano_tensor, torch_tensor, atol=atol, rtol=rtol)
    check_grad(nano_tensor, torch_tensor, atol=atol_grad, rtol=rtol_grad)

def create_identical_torch_tensor(*args):
    torch_tensors = []
    for arg in args:
        t = torch.tensor(arg.data, requires_grad=arg.requires_grad, dtype=torch.float32)
        torch_tensors.append(t)
    return tuple(torch_tensors) if len(torch_tensors) > 1 else torch_tensors[0]


#######################
####### TEST OPS ######
#######################

def make_test_ops(shapes, fcn_nanograd, fcn_torch=None,test_backward:bool=True, discrete=False, 
                  atol:float=1e-6, rtol:float=1e-3, atol_grad:float=1e-6, rtol_grad:float=1e-3, device=Device.CPU, name=None):

    np.random.seed(0)
    torch.manual_seed(0)

    profiler = OpProfiler(func_name=name, device=device)
    fcn_torch = fcn_nanograd if fcn_torch is None else fcn_torch

    tensors = []
    pytorch_tensors = []
    for shape in shapes:
        if discrete:
            t = Tensor.randint(0, 10, shape, requires_grad=test_backward)
            pt = torch.tensor(t.data, requires_grad=test_backward, dtype=torch.int32)
        else:
            t = Tensor.normal(30, 1, shape, requires_grad=test_backward)
            pt = torch.tensor(t.data, requires_grad=test_backward, dtype=torch.float32)

        if device == Device.GPU: 
            t.gpu()

        tensors.append(t)
        pytorch_tensors.append(pt)

    with profiler('nano_fp'):
        out = fcn_nanograd(*tensors)
    with profiler('torch_fp'):
        out_torch = fcn_torch(*pytorch_tensors)

    if test_backward:
        with profiler('nano_bp'):
            out.mean().backward()
        with profiler('torch_bp'):
            out_torch.mean().backward()
    
    profiler.print()

    if device == Device.GPU:
        out.cpu()
        tensors = [t.cpu() for t in tensors]
    
    check_val(out, out_torch, atol=atol, rtol=rtol)

    if test_backward:
        for tensor, pytorch_tensor in zip(tensors, pytorch_tensors):
            check_grad(tensor, pytorch_tensor, atol=atol_grad, rtol=rtol_grad)
    

#######################
##### TEST MODULE #####
#######################

def get_same_pytorch_model(model):
    layers = []
    
    for l in model.layers:
        if isinstance(l, nnn.Linear):
            in_f, out_f = l.in_features, l.out_features
            weight = torch.Tensor(l.weight.data)
            torch_layer = nn.Linear(in_f, out_f)
            torch_layer.weight = nn.Parameter(weight)
            if hasattr(l, "bias"): 
                bias = torch.Tensor(l.bias.data)
                torch_layer.bias = nn.Parameter(bias)
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
            kernel_size = l.kernel_size
            weight, bias = torch.Tensor(l.weight.data), torch.Tensor(l.bias.data)
            torch_layer = (nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding) if isinstance(l, nnn.Conv1d) 
                          else nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding))
            torch_layer.weight, torch_layer.bias = nn.Parameter(weight), nn.Parameter(bias)
            layers.append(torch_layer)
        
        elif isinstance(l, (nnn.MaxPool1d, nnn.AvgPool1d)):
            size = l.pool_size
            torch_layer = nn.MaxPool1d(size) if isinstance(l, nnn.MaxPool1d) else nn.AvgPool1d(size)
            layers.append(torch_layer)

        elif isinstance(l, (nnn.MaxPool2d, nnn.AvgPool2d)):
            size = l.pool_size
            torch_layer = nn.MaxPool2d(size) if isinstance(l, nnn.MaxPool2d) else nn.AvgPool2d(size)
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

def get_same_pytorch_criterion(criterion):
    if isinstance(criterion, nnn.NLLLoss):
        return nn.NLLLoss()
    elif isinstance(criterion, nnn.MSELoss):
        return nn.MSELoss()
    else:
        raise NotImplementedError("Not supported Nanograd criterion")

def make_test_module(shape_inp, shape_target, model, atol=1e-5, atol_grad=1e-5, rtol=1e-3, rtol_grad=1e-3, device=Device.CPU):
    np.random.seed(0)
    torch.manual_seed(0)

    inp = Tensor.normal(30, 2, shape_inp, requires_grad=True)
    target = Tensor.normal(30, 1, shape_target, requires_grad=True)
    inp_torch, targ_torch = create_identical_torch_tensor(inp, target)

    model_torch = get_same_pytorch_model(model)

    if device == Device.GPU:
        inp.gpu()
        model.gpu()
    
    ret = model(inp)
    ret_torch = model_torch(inp_torch)

    ret.mean().backward()
    ret_torch.mean().backward()

    if device == Device.GPU:
        ret.cpu()
        model.cpu()

    check_val(ret, ret_torch, atol=atol, rtol=rtol)
    check_model_parameters(model, model_torch, atol=atol, atol_grad=atol_grad, rtol=rtol, rtol_grad=rtol_grad)

def check_model_parameters(ng_model, pytorch_model, atol=1e-5, atol_grad=1e-4, rtol=1e-7, rtol_grad=1e-5):
    pytorch_layers = [module for module in pytorch_model.modules() if type(module) != nn.Sequential]
    for ng_l, pt_l in zip(ng_model.layers, pytorch_layers):
        if isinstance(ng_l, nnn.Linear):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol, rtol=rtol)
            if hasattr(ng_l, "bias"):
                np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol, rtol=rtol)
            if ng_l.weight.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)
        
        elif isinstance(ng_l, (nnn.BatchNorm1d, nnn.BatchNorm2d)):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.running_mean.data, pt_l.running_mean.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.running_var.data, pt_l.running_var.detach().numpy(), atol=atol)

            if ng_l.weight.grad and ng_l.bias.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)
        
        elif isinstance(ng_l, (nnn.Conv1d, nnn.Conv2d)):
            np.testing.assert_allclose(ng_l.weight.data, pt_l.weight.detach().numpy(), atol=atol)
            np.testing.assert_allclose(ng_l.bias.data, pt_l.bias.detach().numpy(), atol=atol)

            if ng_l.weight.grad and ng_l.bias.grad:
                np.testing.assert_allclose(ng_l.weight.grad.data, pt_l.weight.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)
                np.testing.assert_allclose(ng_l.bias.grad.data, pt_l.bias.grad.detach().numpy(), atol=atol_grad, rtol=rtol_grad)


#######################
###### TEST MNIST #####
#######################

def train(model, X, y, optimizer, steps=1000, batch_size=128, criterion=nnn.NLLLoss, device=Device.CPU):
    model.train()
    losses, accuracies = [], []

    if device == Device.GPU: 
        model.gpu()

    t = trange(steps, desc="Steps")
    for step in t:
        sample = np.random.randint(0, X.shape[0], size=(batch_size))

        Xb = Tensor(X[sample], device=device)
        Yb = Tensor(y[sample], device=device)

        out = model(Xb)

        optimizer.zero_grad()

        loss = criterion()(out, Yb)
        loss.backward()

        optimizer.step()

        cat = np.argmax(out.cpu().data, axis=-1)
        accuracy = (cat == Yb.cpu().data).mean()

        loss = loss.cpu().data
        losses.append(loss)
        accuracies.append(accuracy)

        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model, X, y, batch_size=128, num_classes=10, device=Device.CPU):
    model.eval()
    Y_pred = np.zeros((y.shape[0], num_classes))

    if device == Device.GPU: model.gpu()

    t = trange((len(y) - 1) // batch_size + 1)
    for i in t:
        Xb = Tensor(X[batch_size * i : batch_size * (i+1), :])
        if device == Device.GPU: Xb.gpu()
        Y_pred[batch_size * i : batch_size * (i+1), :] = model(Xb).cpu().data
    
    Y_pred = Y_pred.argmax(-1)
    
    acc = (y == Y_pred).mean()
    print('test set accuracy is %f' % acc)
    return acc
    