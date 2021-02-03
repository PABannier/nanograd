from tests.helpers import check_val_and_grad, create_identical_torch_tensor
from tests.helpers import get_same_pytorch_model, check_model_parameters

from nanograd.tensor import Tensor
import nanograd.nn.module as nnn

import numpy as np    

def test_linear():
    inp = Tensor.normal(0, 1, (32, 10), requires_grad=True)
    model = nnn.Sequential(nnn.Linear(10, 30))

    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_batchnorm_1d():
    inp = Tensor.normal(0, 1, (32, 10), requires_grad=True)
    model = nnn.Sequential(nnn.BatchNorm1d(10))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_batchnorm_2d():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.BatchNorm2d(3))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_conv_1d_no_padding():
    inp = Tensor.normal(0, 1, (5, 3, 10), requires_grad=True)
    model = nnn.Sequential(nnn.Conv1d(3, 10, 3, 2), nnn.Conv1d(10, 32, 2, 2))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_conv_1d_with_padding():
    inp = Tensor.normal(0, 1, (5, 3, 10), requires_grad=True)
    model = nnn.Sequential(nnn.Conv1d(3, 10, 3, 2, 3), nnn.Conv1d(10, 32, 2, 2, 3))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_conv_2d_no_padding():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 2), nnn.Conv2d(10, 32, 3, 3))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model)

def test_conv_2d_with_padding():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.Conv2d(3, 10, 3, 2, 3), nnn.Conv2d(10, 32, 3, 3, 2))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(y, y_torch)
    check_val_and_grad(inp, inp_torch)
    check_model_parameters(model, pytorch_model, atol=1e-4)

def test_maxpool_1d():
    inp = Tensor.normal(0, 1, (5, 3, 10), requires_grad=True)
    model = nnn.Sequential(nnn.MaxPool1d(3, stride=3))

    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_avgpool_1d():
    inp = Tensor.normal(0, 1, (5, 3, 10), requires_grad=True)
    model = nnn.Sequential(nnn.AvgPool1d(3, stride=3))

    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_maxpool_2d():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.MaxPool2d((2, 2)))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_avgpool_2d():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.AvgPool2d((2, 2)))
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_flatten():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.Flatten())
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_relu():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.ReLU())
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_leaky_relu():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.LeakyReLU())
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_sigmoid():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.Sigmoid())
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)

def test_tanh():
    inp = Tensor.normal(0, 1, (5, 3, 7, 7), requires_grad=True)
    model = nnn.Sequential(nnn.Tanh())
    
    inp_torch = create_identical_torch_tensor(inp)
    pytorch_model = get_same_pytorch_model(model)

    y = model(inp)
    y_torch = pytorch_model(inp_torch)

    y.backward()
    y_torch.sum().backward()

    check_val_and_grad(inp, inp_torch)
    check_val_and_grad(y, y_torch)
