from tensor import Tensor
import nn.functional as F

import numpy as np

class Module:
    r"""
        Base class (superclass) for all components of an NN.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html

        Layer classes and even full Model classes should inherit from this Module.
        Inheritance gives the subclass all the functions/variables below
    """

    def __init__(self):
        self._submodules = {} # Submodules of the class
        self._parameters = {} # Trainable parameters in modules and its submodules

        self.is_train = True
    
    def train(self):
        r"""Activates training mode for network component"""
        self.is_train = True

    def eval(self):
        r"""Activates evaluation mode for network component"""
        self.is_train = False
    
    def forward(self, *args):
        r"""Forward pass of the module"""
        raise NotImplementedError("Subclasses of module should have a forward method implemented")

    def is_parameter(self, obj):
        r"""Checks if input object is a Tensor of trainable params"""
        return isinstance(obj, Tensor) and obj.is_parameter

    def parameters(self):
        r"""
            Returns an interator over stored params.
            Includes submodules' params too
        """
        self._ensure_is_initialized()
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter
    
    def add_parameter(self, name, value):
        r"""Stores params"""
        self._ensure_is_initialized()
        self._parameters[name] = value

    def add_module(self, name, value):
        r"""Stores module and its params"""
        self._ensure_is_initialized()
        self._submodules[name] = value
    
    def __setattr__(self, name, value):
        r"""Stores params or modules that you provide"""
        if self.is_parameter(value):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)

        object.__setattr__(self, name, value)

    def __call__(self, *args):
        """Runs self.forward(args)"""
        return self.forward(*args)

    def _ensure_is_initialized(self):
        """Ensures that subclass's __init__() method ran super().__init__()"""
        if self.__dict__.get('_submodules') is None:
            raise Exception("Module not intialized. "
                            "Did you forget to call super().__init__()?")


class Sequential(Module):
    r"""
        Passes input data through stored layers, in order

        >>> model = Sequential(Linear(2,3), ReLU())
        >>> model(x)
        <output after linear then relu>

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        for idx, l in enumerate(self.layers):
            self.add_module(str(idx), l)

    def __iter__(self):
        r"""Enables list-like iteration through layers"""
        yield from self.layers

    def __getitem__(self, idx):
        r"""Enables list-like indexing for layers"""
        return self.layers[idx]

    def train(self):
        r"""Sets this object and all trainable modules within to train mode"""
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self):
        r"""Sets this object and all trainable modules within to eval mode"""
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x):
        r"""
            Passes input data through each layer in order
            
            Args:
                x (Tensor): Input data
            Returns:
                Tensor: Output after passing through layers
        """
        for layer in self.layers: 
            x = layer(x)
        return x


class Linear(Module):
    r"""
        A linear layer (aka 'fully-connected' or 'dense' layer)

        >>> layer = Linear(2,3)
        >>> layer(Tensor.ones(10,2)) # (batch_size, in_features)
        <some tensor output with size (batch_size, out_features)>

        Args:
            in_features (int): # dims in input
                            (i.e. # of inputs to each neuron)
            out_features (int): # dims of output
                            (i.e. # of neurons)

        Inherits from:
            Module (mytorch.nn.module.Module)
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Randomly initializing layer weights
        k = 1 / in_features
        weight = k * (np.random.rand(out_features, in_features) - 0.5)
        bias = k * (np.random.rand(out_features) - 0.5)
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_features)
        Returns:
            Tensor: (batch_size, out_features)
        """
        
        return x @ self.weight.T() + self.bias


class BatchNorm1d(Module):
    """Batch Normalization Layer

        Args:
            num_features (int): # dims in input and output
            eps (float): value added to denominator for numerical stability
                        (not important for now)
            momentum (float): value used for running mean and var computation

        Inherits from:
            Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(eps)
        self.momentum = Tensor(momentum)

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        if self.is_train == True:
            batch_mean = x.mean(0)
            batch_var = ((x - batch_mean) ** 2).mean(0)
            batch_empirical_var = ((x - batch_mean) ** 2).sum(0) / Tensor(x.shape[0] - 1)

            self.running_mean = (Tensor(1) - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (Tensor(1) - self.momentum) * self.running_var + self.momentum * batch_empirical_var

            return self._normalize(x, batch_mean, batch_var)
        else:
            return self._normalize(x, self.running_mean, self.running_var)
        
    def _normalize(self, x, mean, var):
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta


# ***** Activation functions *****


class ReLU(Module):
    r"""
        ReLU Activation Layer

        Applies a Rectified Linear Unit activation function to 
        the input

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return F.ReLU.apply(x)


class Sigmoid(Module):
    r"""
        Sigmoid Activation Layer

        Applies a Sigmoid activation function to the input

        Inherits from:
            Module (nn.module.Modue)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x)
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return F.Sigmoid.apply(x)


class Tanh(Module):
    r"""
        Tanh Activation Layer

        Applies a Hyperbolic Tangent activation function to the input

        Inherits from:
            Module (nn.module.Modue)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x)
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return F.Tanh.apply(x)