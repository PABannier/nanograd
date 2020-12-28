import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensor import Tensor
from nn.module import BatchNorm1d, BatchNorm2d, Linear, ReLU, Sequential, Conv1d, Conv2d, MaxPool2d, Flatten
from nn.loss import CrossEntropyLoss
from optim.optimizer import SGD

MNIST_PATHS = [
    '../data/train-images-idx3-ubyte.gz',
    '../data/train-labels-idx1-ubyte.gz',
    '../data/t10k-images-idx3-ubyte.gz',
    '../data/t10k-labels-idx1-ubyte.gz'
]

def load_mnist():
    print("Loading data...")
    import gzip
    mnist = []
    for path in MNIST_PATHS:
        with open(path, 'rb') as f:
            dat = f.read()
            arr = np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
            mnist.append(arr)
    
    return tuple(mnist)


def check_val(nano_tensor, torch_tensor, atol=1e-5):
    assert np.allclose(nano_tensor.data, torch_tensor.data.numpy(), atol=atol)


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
    assert type(nano_tensor).__name__ == "Tensor", f"Expected Tensor object, got {type(nano_tensor).__name__}"
    check_val(nano_tensor, torch_tensor)
    check_grad(nano_tensor, torch_tensor)


def create_identical_torch_tensor(*args):
    torch_tensors = []
    for arg in args:
        t = torch.tensor(arg.data.astype(np.float32), requires_grad=arg.requires_grad)
        torch_tensors.append(t)
    return tuple(torch_tensors) if len(torch_tensors) > 1 else torch_tensors[0]


def get_mytorch_model_input_features(model):
    """
    Returns in_features for the first linear layer of a
    Sequential model.
    """
    return get_mytorch_linear_layers(model)[0].in_features


def get_mytorch_model_output_features(model):
    """
    Returns out_features for the last linear layer of a
    Sequential model.
    """
    return get_mytorch_linear_layers(model)[-1].out_features


def get_mytorch_linear_layers(model):
    """
    Returns a list of linear layers for a model.
    """
    return list(filter(lambda x: isinstance(x, Linear), model.layers))


def get_pytorch_linear_layers(pytorch_model):
    """
    Returns a list of linear layers for a pytorch model.
    """
    return list(filter(lambda x: isinstance(x, nn.Linear), pytorch_model))


def get_same_pytorch_mlp(model):
    """
    Returns a pytorch Sequential model matching the given mytorch mlp, with
    weights copied over.
    """
    layers = []
    for l in model.layers:
        if isinstance(l, Linear):
            layers.append(nn.Linear(l.in_features, l.out_features))
            layers[-1].weight = nn.Parameter(
                torch.tensor(l.weight.data).double())
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data).double())

        elif isinstance(l, BatchNorm1d):
            layers.append(nn.BatchNorm1d(int(l.num_features)))
            layers[-1].weight = nn.Parameter(
                torch.tensor(l.gamma.data).double())
            layers[-1].bias = nn.Parameter(torch.tensor(l.beta.data).double())

        elif isinstance(l, BatchNorm2d):
            layers.append(nn.BatchNorm2d(l.size))
            layers[-1].weight = nn.Parameter(
                torch.tensor(l.gamma.data).double())
            layers[-1].bias = nn.Parameter(torch.tensor(l.beta.data).double())

        elif isinstance(l, ReLU):
            layers.append(nn.ReLU())

        elif isinstance(l, Flatten):
            layers.append(nn.Flatten())

        elif isinstance(l, Conv1d):
            layers.append(nn.Conv1d(int(l.in_channel), int(l.out_channel), \
                                    l.kernel_size, int(l.stride), int(l.padding)))
            layers[-1].weight = nn.Parameter(
                torch.tensor(l.weight.data).double())
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data).double())

        elif isinstance(l, Conv2d):
            layers.append(nn.Conv2d(int(l.in_channel), int(l.out_channel), \
                                    l.kernel_size, int(l.stride), int(l.padding)))
            layers[-1].weight = nn.Parameter(
                torch.tensor(l.weight.data).double())
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data).double())
        
        elif isinstance(l, MaxPool2d):
            layers.append(nn.MaxPool2d(l.kernel_size, l.stride))

        else:
            raise Exception("Unrecognized layer in Nanograd model")
    pytorch_model = nn.Sequential(*layers)
    return pytorch_model.double()


def get_same_pytorch_optimizer(optimizer, pytorch_mlp):
    """
    Returns a pytorch optimizer matching the given mytorch optimizer, except
    with the pytorch mlp parameters, instead of the parametesr of the mytorch
    mlp
    """
    lr = optimizer.lr
    momentum = optimizer.momentum
    return torch.optim.SGD(pytorch_mlp.parameters(), lr=lr, momentum=momentum)


def get_same_pytorch_criterion(criterion):
    """
    Returns a pytorch criterion matching the given mytorch optimizer
    """
    if criterion is None:
        return None
    return nn.CrossEntropyLoss()


def generate_dataset_for_mytorch_model(model, batch_size):
    """
    Generates a fake dataset to test on.

    Returns x: ndarray (batch_size, in_features),
            y: ndarray (batch_size,)
    where in_features is the input dim of the mytorch_model, and out_features
    is the output dim.
    """
    in_features = get_mytorch_model_input_features(model)
    out_features = get_mytorch_model_output_features(model)
    x = np.random.randn(batch_size, in_features)
    y = np.random.randint(out_features, size=(batch_size,))
    return x, y


def forward_(mytorch_model, mytorch_criterion, pytorch_model,
             pytorch_criterion, x, y):
    """
    Calls forward on both mytorch and pytorch models.

    x: ndrray (batch_size, in_features)
    y: ndrray (batch_size,)

    Returns (passed, (mytorch x, mytorch y, pytorch x, pytorch y)),
    where passed is whether the test passed
    """
    # forward
    pytorch_x = Variable(torch.tensor(x).double(), requires_grad=True)
    pytorch_y = pytorch_model(pytorch_x)
    if not pytorch_criterion is None:
        pytorch_y = pytorch_criterion(pytorch_y, torch.LongTensor(y))

    mytorch_x = Tensor(x, requires_grad=True)
    mytorch_y = mytorch_model(mytorch_x)
    if not mytorch_criterion is None:
        mytorch_y = mytorch_criterion(mytorch_y, Tensor(y))

    # forward check
    assert assertions_all(mytorch_y.data, pytorch_y.detach().numpy(), 'y'), "Forward Failed"
    check_model_param_settings(mytorch_model)

    return (mytorch_x, mytorch_y, pytorch_x, pytorch_y)


def backward_(mytorch_x, mytorch_y, mytorch_model, pytorch_x, pytorch_y, pytorch_model):
    """
    Calls backward on both mytorch and pytorch outputs, and returns whether
    computed gradients match.
    """
    mytorch_y.backward()
    pytorch_y.sum().backward()
    check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model)
    check_model_param_settings(mytorch_model)


def check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model):
    """
    Checks computed gradients, assuming forward has already occured.

    Checked gradients are the gradients of linear weights and biases, and the
    gradient of the input.
    """

    assert assertions_all(mytorch_x.grad.data, pytorch_x.grad.detach().numpy(), 'dx'), "Gradient Check Failed"

    mytorch_linear_layers = get_mytorch_linear_layers(mytorch_model)
    pytorch_linear_layers = get_pytorch_linear_layers(pytorch_model)
    for mytorch_linear, pytorch_linear in zip(mytorch_linear_layers, pytorch_linear_layers):
        pytorch_dW = pytorch_linear.weight.grad.detach().numpy()
        pytorch_db = pytorch_linear.bias.grad.detach().numpy()
        mytorch_dW = mytorch_linear.weight.grad.data
        mytorch_db = mytorch_linear.bias.grad.data

        assert assertions_all(mytorch_dW, pytorch_dW, 'dW'), "Gradient Check Failed"
        assert assertions_all(mytorch_db, pytorch_db, 'db'), "Gradient Check Failed"


def assertions_all(user_vals, expected_vals, test_name, rtol=1e-5, atol=1e-8):
    if not assertions(user_vals, expected_vals, 'type', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'shape', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'closeness', test_name, rtol=rtol, atol=atol):
        return False
    return True


def assertions(user_vals, expected_vals, test_type, test_name, rtol=1e-5, atol=1e-8):
    if test_type == 'type':
        try:
            assert type(user_vals) == type(expected_vals)
        except Exception as e:
            print('Type error, your type doesnt match the expected type.')
            print('Wrong type for %s' % test_name)
            print('Your type:   ', type(user_vals))
            print('Expected type:', type(expected_vals))
            return False
    elif test_type == 'shape':
        try:
            assert user_vals.shape == expected_vals.shape
        except Exception as e:
            print('Shape error, your shapes doesnt match the expected shape.')
            print('Wrong shape for %s' % test_name)
            print('Your shape:    ', user_vals.shape)
            print('Expected shape:', expected_vals.shape)
            return False
    elif test_type == 'closeness':
        try:
            assert np.allclose(user_vals, expected_vals, rtol=rtol, atol=atol)
        except Exception as e:
            print('Closeness error, your values dont match the expected values.')
            print('Wrong values for %s' % test_name)
            print('Your values:    ', user_vals)
            print('Expected values:', expected_vals)
            return False
    return True


def check_param_tensor(param):
    """Runs various (optional, ungraded) tests that confirm whether model param tensors are correctly configured

    Note: again these tests aren't graded, although they will be next semester.
    
    Args:
        param (Tensor): Parameter tensor from model
    """
    assert type(param).__name__ == 'Tensor', f"The param must be a tensor. You likely replaced a module param tensor on accident.\n\tCurrently: {type(param).__name__}, Expected: Tensor"
    assert isinstance(param.data, np.ndarray), f"The param's .data must be a numpy array (ndarray). You likely put a tensor inside another tensor somewhere.\n\tCurrently: {type(param.data).__name__}, Expected: ndarray"

    assert param.is_parameter == True, f"The param must have is_parameter==True.\n\tCurrently: {param.is_parameter}, Expected: True"
    assert param.requires_grad == True, f"The param must have requires_grad==True.\n\tCurrently: {param.requires_grad}, Expected: True"
    assert param.is_leaf == True, f"The param must have is_leaf==True.\n\tCurrently: {param.is_leaf}, Expected: True"

    if param.grad is not None:
        assert type(param.grad).__name__ == 'Tensor', f"If a module tensor has a gradient, the gradient MUST be a Tensor\n\tCurrently: {type(param).__name__}, Expected: Tensor"
        assert param.grad.grad is None, f"Gradient of module parameter (weight or bias tensor) must NOT have its own gradient\n\tCurrently: {param.grad.grad}, Expected: None"
        assert param.grad.grad_fn is None, f"Gradient of module parameter (weight or bias tensor) must NOT have its own grad_fn\n\tCurrently: {param.grad.grad_fn}, Expected: None"
        assert param.grad.is_parameter == False, f"Gradient of module parameter should NOT have is_parameter == True.\n\tCurrently: {param.is_parameter}, Expected: False"
        assert param.grad.shape == param.shape, f"The gradient tensor of a parameter must have the same shape as the parameter\n\tCurrently: {param.grad.shape}, Expected: {param.shape}"


def check_model_param_settings(model):
    """Checks that the parameters of a model are correctly configured.
    
    Note: again these tests aren't graded, although they will be next semester.

    Args:
        model (mytorch.nn.sequential.Sequential) 
    """
    # Iterate through layers and perform checks for each layer
    for idx, l in enumerate(model.layers):
        # Check that weights and biases of linear and conv1d layers are correctly configured
        if type(l).__name__ in ["Linear", "Conv1d"]:
            try:
                check_param_tensor(l.weight)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has parameter (weight) tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is to help you debug.")
                return False

            try:
                check_param_tensor(l.bias)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has parameter (bias) tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
        
        # Check batchnorm is correct
        elif type(l).__name__ == "BatchNorm1d":
            try:
                check_param_tensor(l.gamma)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has gamma tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
            
            try:
                check_param_tensor(l.beta)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has beta tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
            
            try:
                assert type(l.running_mean).__name__ == 'Tensor', f"Running mean param of BatchNorm1d must be a tensor. \n\tCurrently: {type(l.running_mean).__name__}, Expected: Tensor"
                assert type(l.running_var).__name__ == 'Tensor', f"Running var param of BatchNorm1d must be a tensor. \n\tCurrently: {type(l.running_var).__name__}, Expected: Tensor"
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has running mean/var tensors with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
        # TODO: Check that weights and biases of LSTM layers are correctly configured
        
    return True


def forward_test(model, criterion=None, batch_size=(2,5)):
    """
        Tests forward, printing whether a mismatch occurs in forward.
    """
    pytorch_model = get_same_pytorch_mlp(model)
    batch_size = np.random.randint(*batch_size) if type(batch_size) == tuple\
        else batch_size
    x, y = generate_dataset_for_mytorch_model(model, batch_size)
    pytorch_criterion = get_same_pytorch_criterion(criterion)

    forward_(model, criterion, pytorch_model, pytorch_criterion, x, y)


def forward_backward_test(model, criterion=None, batch_size=(2,5)):
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


def step_test(model, optimizer, train_steps, eval_steps,
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