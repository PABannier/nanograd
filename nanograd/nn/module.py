from tensor import Tensor
import nn.functional as F

import numpy as np


def init_weights(tensor, weight_initialization, fan_mode="fan_in"):
    assert isinstance(tensor, Tensor), f"Only Tensor objects are accepted. Got {type(layer).__name__}"
    assert fan_mode in ["fan_in", "fan_out"], "Wrong fan mode. Only fan_in and fan_out supported."

    out_features, in_features = tensor.shape[0], tensor.shape[1]

    if weight_initialization == "kaiming_normal":
        std = np.sqrt(1 / in_features) if fan_mode == "fan_in" else np.sqrt(1.0 / out_features)
        return Tensor.normal(0.0, std, tensor.shape, requires_grad=tensor.requires_grad, \
                                                     is_parameter=tensor.is_parameter)
    elif weight_initialization == "kaiming_uniform":
        bound = np.sqrt(3 / in_features) if fan_mode == "fan_in" else np.sqrt(3.0 / out_features)
        weight = np.random.uniform(-bound, bound, tensor.shape)
        return Tensor(weight, requires_grad=tensor.requires_grad, \
                              is_parameter=not tensor.requires_grad)
    elif weight_initialization == "glorot_normal":
        std = np.sqrt(2.0 / (in_features + out_features))
        return Tensor.normal(0.0, std, tensor.shape, requires_grad=tensor.requires_grad, \
                                                     is_parameter=tensor.is_parameter)
    elif weight_initialization == "glorot_uniform":
        bound = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-bound, bound, tensor.shape) 
        return Tensor(weight, requires_grad=tensor.requires_grad, \
                              is_parameter=not tensor.requires_grad)
    else:
        raise Exception("Unknown weight initialization methods. Only Glorot and Kaiming are available.")


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
    def __init__(self, in_features, out_features, 
                 weight_initialization="kaiming_normal", fan_mode="fan_in"):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor.zeros((self.out_features, self.in_features), requires_grad=True, is_parameter=True)
        self.weight = init_weights(self.weight, weight_initialization, fan_mode)
        self.bias = Tensor.zeros(self.out_features, requires_grad=True, is_parameter=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_features)
        Returns:
            Tensor: (batch_size, out_features)
        """
        return x @ self.weight.T() + self.bias        


class BatchNorm1d(Module):
    """Batch Normalization Layer 1d

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


class BatchNorm2d(Module):
    """Batch Normalization Layer 2d

        Args:
            num_features (int): # dims in input and output
            eps (float): value added to denominator for numerical stability
                        (not important for now)
            momentum (float): value used for running mean and var computation

        Inherits from:
            Module (mytorch.nn.module.Module)
    """
    def __init__(self, size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.size = size

        self.eps = Tensor(eps)
        self.momentum = Tensor(momentum)

        # To make the final output affine
        self.gamma = Tensor.ones(self.size, requires_grad=True, is_parameter=True)
        self.beta = Tensor.zeros(self.size, requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor.zeros(self.size, requires_grad=False, is_parameter=False)
        self.running_var = Tensor.ones(self.size, requires_grad=False, is_parameter=False)

    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        if self.is_train == True:
            batch_mean = x.mean(axis=(0, 2, 3))
            batch_var = ((x - batch_mean.reshape(shape=[1, -1, 1, 1])) ** 2).mean(axis=(0, 2, 3))
            batch_empirical_var = ((x - batch_mean.reshape(shape=[1, -1, 1, 1])) ** 2).sum(axis=(0, 2, 3)) / Tensor(x.shape[0] - 1)

            self.running_mean = (Tensor(1) - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (Tensor(1) - self.momentum) * self.running_var + self.momentum * batch_empirical_var

            return self._normalize(x, batch_mean, batch_var)
        else:
            return self._normalize(x, self.running_mean, self.running_var)
        
    def _normalize(self, x, mean, var):
        x_hat = (x - mean.reshape(shape=[1, -1, 1, 1])) / (var + self.eps).sqrt().reshape(shape=[1, -1, 1, 1])
        return self.gamma.reshape(shape=[1, -1, 1, 1]) * x_hat + self.beta.reshape(shape=[1, -1, 1, 1])


class Conv1d(Module):
    r"""
        1-dimensional convolutional layer.
       
        Args:
            in_channel (int): # channels in input (example: # color channels in image)
            out_channel (int): # channels produced by layer
            kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
            stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                       weight_initialization="kaiming_normal"):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.stride, self.padding = stride, padding
        self.kernel_size = kernel_size
        self.weight_initialization = weight_initialization

        shape = (self.out_channel, self.in_channel, self.kernel_size)
        self.weight = Tensor.zeros(shape, requires_grad=True, is_parameter=True)
        self.weight = init_weights(self.weight, self.weight_initialization)
        self.bias = Tensor.zeros(out_channel, requires_grad=True, is_parameter=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_size)
        Returns:
            Tensor: (batch_size, out_channel, output_size)
        """
        return F.Conv1d.apply(x, self.weight, self.bias, self.stride, self.padding)


class Conv2d(Module):
    r"""
        2-dimensional convolutional layer.

        Args:
            in_channel (int): # channels in input (example: # color channels in image)
            out_channel (int): # channels produced by layer
            kernel_size (tuple): edge lengths of the kernel
            stride (int): stride of the convolution (filter)
            padding (int): padding for the convolution
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                       weight_initialization="kaiming_normal"):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride, self.padding = stride, padding
        self.weight_initialization = weight_initialization

        shape = (self.out_channel, self.in_channel, *self.kernel_size)
        self.weight = Tensor.zeros(shape, requires_grad=True, is_parameter=True)
        self.weight = init_weights(self.weight, self.weight_initialization)
        self.bias = Tensor.zeros(self.out_channel, requires_grad=True, is_parameter=True)
    
    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, in_channel, width, height)
            Returns:
                Tensor: (batch_size, out_channel, output_dim1, output_dim2)
        """
        return F.Conv2d.apply(x, self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    r"""
        Performs a max pooling operation after a 2d convolution 
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
    
    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, channel, in_width, in_height)
            
            Returns:
                Tensor: (batch_size, channel, out_width, out_height)
        """
        return F.MaxPool2d.apply(x, self.kernel_size, self.stride)


class Flatten(Module):
    r"""
        Layer that flattens all dimensions for each observation in a batch

        >>> x = torch.randn(4, 3, 2) # batch of 4 observations, each sized (3, 2)
        >>> x
        tensor([[[ 0.8816,  0.9773],
                [-0.1246, -0.1373],
                [-0.1889,  1.6222]],

                [[-0.9503, -0.8294],
                [ 0.8900, -1.2003],
                [-0.9701, -0.4436]],

                [[ 1.7809, -1.2312],
                [ 1.0769,  0.6283],
                [ 0.4997, -1.7876]],

                [[-0.5303,  0.3655],
                [-0.7496,  0.6935],
                [-0.8173,  0.4346]]])
        >>> layer = Flatten()
        >>> out = layer(x)
        >>> out
        tensor([[ 0.8816,  0.9773, -0.1246, -0.1373, -0.1889,  1.6222],
                [-0.9503, -0.8294,  0.8900, -1.2003, -0.9701, -0.4436],
                [ 1.7809, -1.2312,  1.0769,  0.6283,  0.4997, -1.7876],
                [-0.5303,  0.3655, -0.7496,  0.6935, -0.8173,  0.4346]])
        >>> out.shape
        torch.size([4, 6]) # batch of 4 observations, each flattened into 1d array size (6,)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        r"""
            Args:
                x (Tensor): (batch_size, dim_2, dim_3, ...) arbitrary number of dims after batch_size
            Returns:
                out (Tensor): (batch_size, dim_2 * dim_3 * ...) batch_size, then all other dims flattened
        """
        dim1 = x.shape[0]
        dim2 = np.prod(x.shape[1:])
        return x.reshape((dim1, dim2))


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.mask, self.p = None, p
    
    def forward(self, x):
        if self.is_train:
            if self.mask is None:
                val = np.random.binomial(1, self.p, size=x.shape)
                self.mask = Tensor(val)
            return x * self.mask
        
        return x


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
    
    def forward(self, x):
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
    
    def forward(self, x):
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return F.Tanh.apply(x)