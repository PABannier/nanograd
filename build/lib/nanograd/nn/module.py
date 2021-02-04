from nanograd.tensor import Tensor
import nanograd.tensor as tensor
from nanograd.device import Device

import numpy as np

from typing import Union


def init_weights(shape:tuple, weight_initialization:str, 
                 fan_mode:str="fan_in", **kwargs) -> Tensor:
    """Initializes the weights of a layer

    Args:
        shape (tuple): Shape of the tensor
        weight_initialization (str): Method of initialization (Glorot or Kaiming)
        fan_mode (str, optional): Fan mode. Defaults to "fan_in".

    Raises:
        Exception: Method must be Kaiming or Xavier normal or uniform 

    Returns:
        Tensor: Tensor containing the initialized weights
    """
    assert type(shape) == tuple, f"Shape must be a tuple. Got {type(shape).__name__}."
    assert fan_mode in ["fan_in", "fan_out"], "Wrong fan mode. Only fan_in and fan_out supported."

    out_features, in_features = shape[0], shape[1]

    if weight_initialization == "kaiming_normal":
        std = np.sqrt(1.0 / in_features) if fan_mode == "fan_in" else np.sqrt(1.0 / out_features)
        return Tensor.normal(0.0, std, shape, **kwargs)

    elif weight_initialization == "kaiming_uniform":
        bound = np.sqrt(3.0 / in_features) if fan_mode == "fan_in" else np.sqrt(3.0 / out_features)
        weight = np.random.uniform(-bound, bound, shape)
        return Tensor(weight, **kwargs)

    elif weight_initialization == "glorot_normal":
        std = np.sqrt(2.0 / (in_features + out_features))
        return Tensor.normal(0.0, std, shape, **kwargs)

    elif weight_initialization == "glorot_uniform":
        bound = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-bound, bound, shape) 
        return Tensor(weight, **kwargs)

    else:
        raise Exception("Unknown weight initialization methods. Only Glorot and Kaiming are available.")


class Module:
    """Base class (superclass) for all components of an NN.

    Layer classes and even full Model classes should inherit from this Module.
    Inheritance gives the subclass all the functions/variables below

    """

    def __init__(self) -> None:
        self._submodules = {} # Submodules of the class
        self._parameters = {} # Trainable parameters in modules and its submodules
        self._tensors = {}

        self.is_train = True
    
    def train(self) -> None:
        """Activates training mode for network component"""
        self.is_train = True

    def eval(self) -> None:
        """Activates evaluation mode for network component"""
        self.is_train = False
    
    def forward(self, *args):
        """Forward pass of the module"""
        raise NotImplementedError("Subclasses of module should have a forward method implemented")

    def is_parameter(self, obj) -> bool:
        """Checks if input object is a Tensor of trainable params"""
        return isinstance(obj, Tensor) and obj.is_parameter

    def parameters(self):
        """
            Returns an interator over stored params.
            Includes submodules' params too
        """
        self._ensure_is_initialized()
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter
    
    def add_parameter(self, name, value) -> None:
        """Stores params"""
        self._ensure_is_initialized()
        self._parameters[name] = value

    def add_module(self, name, value) -> None:
        """Stores module and its params"""
        self._ensure_is_initialized()
        self._submodules[name] = value
    
    def add_tensor(self, name, value) -> None:
        """Stores tensors"""
        self._ensure_is_initialized()
        self._tensors[name] = value
    
    def __setattr__(self, name, value):
        """Stores params or modules that you provide"""
        if self.is_parameter(value):
            self.add_parameter(name, value)
            self.add_tensor(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, Tensor):
            self.add_tensor(name, value)

        object.__setattr__(self, name, value)

    def __call__(self, *args):
        """Runs self.forward(args)"""
        return self.forward(*args)

    def _ensure_is_initialized(self):
        """Ensures that subclass's __init__() method ran super().__init__()"""
        if self.__dict__.get('_submodules') is None:
            raise Exception("Module not intialized. "
                            "Did you forget to call super().__init__()?")
    
    def cpu(self):
        """Moving all tensors onto the CPU"""
        for tensor in self._tensors.items():
            tensor[1].cpu()
        
    def gpu(self):
        """Moving all tensors onto the GPU"""
        for tensor in self._tensors.items():
            tensor[1].gpu()


class Sequential(Module):
    """Passes input data through stored layers, in order

    >>> model = Sequential(Linear(2,3), ReLU())
    >>> model(x)
    <output after linear then relu>

    Inherits from:
        Module (nn.module.Module)
    """
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = layers

        for idx, l in enumerate(self.layers):
            self.add_module(str(idx), l)

    def __iter__(self):
        """Enables list-like iteration through layers"""
        yield from self.layers

    def __getitem__(self, idx:int):
        """Enables list-like indexing for layers"""
        return self.layers[idx]

    def train(self) -> None:
        """Sets this object and all trainable modules within to train mode"""
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self) -> None:
        """Sets this object and all trainable modules within to eval mode"""
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x:Tensor) -> Tensor:
        """Passes input data through each layer in order
            
            Args:
                x (Tensor): Input data
            Returns:
                Tensor: Output after passing through layers
        """
        for layer in self.layers: 
            x = layer(x)
        return x
    
    def gpu(self):
        for layer in self.layers:
            layer.gpu()
    
    def cpu(self):
        for layer in self.layers:
            layer.cpu()

class Linear(Module):
    """A linear layer (aka 'fully-connected' or 'dense' layer)

        >>> layer = Linear(2,3)
        >>> layer(Tensor.ones(10,2)) # (batch_size, in_features)
        <some tensor output with size (batch_size, out_features)>

        Args:
            in_features (int): # dims in input
                            (i.e. # of inputs to each neuron)
            out_features (int): # dims of output
                            (i.e. # of neurons)
            weight_initialization (str): Weight initialization mode
            fan_mode (str): Fan mode as used in the weight initializer
            with_bias (bool): A Linear layer with or without bias

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, in_features:int, out_features:int, 
                 weight_initialization:str="kaiming_normal", fan_mode:str="fan_in",
                 with_bias=True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = with_bias

        self.weight = init_weights(
            (self.out_features, self.in_features), weight_initialization, fan_mode,
            requires_grad=True, is_parameter=True, name="lin_weight")
        
        if self.with_bias:
            self.bias = Tensor.zeros((self.out_features, ), requires_grad=True, 
                                    is_parameter=True, name="lin_bias")

    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, in_features)
            Returns:
                Tensor: (batch_size, out_features)
        """
        if self.with_bias:
            out = x @ self.weight.T() + self.bias.unsqueeze(1).T()
        else:
            out = x @ self.weight.T()
           
        out.name = "lin_res"
        return out   


class BatchNorm1d(Module):
    """Batch Normalization Layer 1d

        Args:
            num_features (int): # dims in input and output
            eps (float): Value added to denominator for numerical stability
            momentum (float): Value used for running mean and var computation (smoothing parameter)

        Inherits from:
            Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1) -> None:
        super().__init__()
        self.num_features = num_features

        self.eps = eps
        self.momentum = momentum

        self.weight = Tensor.ones((self.num_features,), requires_grad=True, is_parameter=True, name="bn_gamma")
        self.bias = Tensor.zeros((self.num_features,), requires_grad=True, is_parameter=True, name="bn_beta")
        
        self.running_mean = Tensor.zeros((self.num_features,), requires_grad=False, is_parameter=False, name="bn_running_mean")
        self.running_var = Tensor.ones((self.num_features,), requires_grad=False, is_parameter=False, name="bn_running_var")

    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        if self.is_train == True:
            batch_mean = x.mean(0)
            batch_var = ((x - batch_mean.reshape(shape=[1, -1])) ** 2).mean(0)
            batch_empirical_var = ((x - batch_mean.reshape(shape=[1, -1])) ** 2).sum(0) / (x.shape[0] - 1)

            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1. - self.momentum) * self.running_var + self.momentum * batch_empirical_var

            return self._normalize(x, batch_mean, batch_var)
        else:
            return self._normalize(x, self.running_mean, self.running_var)
        
    def _normalize(self, x:Tensor, mean:Tensor, var:Tensor) -> Tensor:
        """Performs the actual normalization operation

        Args:
            x (Tensor): Input
            mean (Tensor): Computed batch mean if training, else running mean for validation
            var (Tensor): Computed batch var if training, else running var for validation

        Returns:
            Tensor: Normalized tensor
        """
        x_hat = (x - mean.reshape(shape=[1, -1])) / (var + self.eps).sqrt().reshape(shape=[1, -1])
        out = self.weight.reshape(shape=[1, -1]) * x_hat + self.bias.reshape(shape=[1, -1])
        out.name = "bn_1d_res"
        return out


class BatchNorm2d(Module):
    """Batch Normalization Layer 2d

        Args:
            size (tuple): # dims in input and output
            eps (float): value added to denominator for numerical stability
            momentum (float): value used for running mean and var computation (smoothing parameter)

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1) -> None:
        super().__init__()
        self.num_features = num_features

        self.eps = eps
        self.momentum = momentum

        self.weight = Tensor.ones(self.num_features, requires_grad=True, is_parameter=True, name="bn_gamma")
        self.bias = Tensor.zeros(self.num_features, requires_grad=True, is_parameter=True, name="bn_beta")

        self.running_mean = Tensor.zeros(self.num_features, requires_grad=False, is_parameter=False, name="bn_running_mean")
        self.running_var = Tensor.ones(self.num_features, requires_grad=False, is_parameter=False, name="bn_running_var")

    def forward(self, x:Tensor) -> Tensor:
        """Forward pass

            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        if self.is_train == True: 
            batch_mean = x.mean(axis=(0, 2, 3)) 
            batch_var = ((x - batch_mean.reshape(shape=[1, -1, 1, 1])) ** 2).mean(axis=(0, 2, 3)) 
            batch_empirical_var = (((x - batch_mean.reshape(shape=[1, -1, 1, 1])) ** 2).sum(axis=(0, 2, 3)) / 
                                  (np.prod([x.shape[i] for i in [0, 2, 3]])-1))

            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1. - self.momentum) * self.running_var + self.momentum * batch_empirical_var

            return self._normalize(x, batch_mean, batch_var)
        else:
            return self._normalize(x, self.running_mean, self.running_var)
        
    def _normalize(self, x:Tensor, mean:Tensor, var:Tensor) -> Tensor:
        """Normalize a Tensor with mean and variance

            Args:
                x (Tensor): Input
                mean (Tensor): Computed batch mean if training, else running mean for validation
                var (Tensor): Computed batch var if training, else running var for validation
            
            Returns:
                Tensor: Normalized tensor
        """
        x_hat = (x - mean.reshape(shape=[1, -1, 1, 1])) / (var + self.eps).sqrt().reshape(shape=[1, -1, 1, 1])
        out = self.weight.reshape(shape=[1, -1, 1, 1]) * x_hat + self.bias.reshape(shape=[1, -1, 1, 1])
        out.name = 'bn_2d_res'
        return out


class Conv1d(Module):
    """1-dimensional convolutional layer.
       
        Args:
            in_channel (int): # channels in input (example: # color channels in image)
            out_channel (int): # channels produced by layer
            kernel_size (int): Edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
            stride (int): Stride of the convolution (filter)
            padding (tuple, int): Amount of zero-padding applied before the convolution
            weight_initialization (str): Weight initialization method
            with_bias (bool): Performs a convolution with or without a bias
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int, stride:int=1, 
                 padding:Union[tuple, int]=(0, 0), weight_initialization:str="kaiming_normal") -> None:
        super().__init__()
        assert isinstance(padding, (tuple, int)), 'Wrong padding type. Must be integer or tuple of integers'
        if isinstance(padding, (int)): padding = (padding, padding)
        assert len(padding) == 2, 'Wrong padding dimensions. Padding must be an integer of a 2-dimensional tuple'

        self.in_channel, self.out_channel = in_channel, out_channel
        self.stride, self.padding = stride, padding
        self.kernel_size = kernel_size
        self.weight_initialization = weight_initialization

        shape = (self.out_channel, self.in_channel, self.kernel_size)
        self.weight = init_weights(shape, self.weight_initialization, requires_grad=True, is_parameter=True, name="conv_weight_1d")
        self.bias = Tensor.zeros(out_channel, requires_grad=True, is_parameter=True, name="conv_bias_1d")

    def forward(self, x:Tensor) -> Tensor:
        """Forward pass

            Args:
                x (Tensor): (batch_size, in_channel, input_length)
            Returns:
                Tensor: (batch_size, out_channel, output_length)
        """
        x_padded = x.pad1d(self.padding)
        return x_padded.conv1d(self.weight, self.stride) + self.bias.reshape(shape=[1, -1, 1])


class Conv2d(Module):
    r"""
        2-dimensional convolutional layer.

        Args:
            in_channel (int): # channels in input (example: # color channels in image)
            out_channel (int): # channels produced by layer
            kernel_size (tuple): Edge lengths of the kernel
            stride (int): Stride of the convolution (filter)
            padding (int): Padding for the convolution
            weight_initialization (str): Weight initialization method
            with_bias (bool): Performs a convolution with or without a bias
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, in_channel:int, out_channel:int, kernel_size:int, stride:int=1, 
                 padding:tuple=(0, 0, 0, 0), weight_initialization:str="kaiming_normal",
                 with_bias:bool=True) -> None:
        super().__init__()
        assert isinstance(padding, (tuple, int)), 'Wrong padding type. Must be integer or tuple of integers'
        if isinstance(padding, (int)): padding = (padding, padding, padding, padding)
        assert len(padding) == 4, 'Wrong padding dimensions'
        self.in_channel, self.out_channel = in_channel, out_channel
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride, self.padding = stride, padding
        self.weight_initialization = weight_initialization
        self.with_bias = with_bias

        shape = (self.out_channel, self.in_channel, *self.kernel_size)

        self.weight = init_weights(shape, self.weight_initialization, requires_grad=True, 
                                   is_parameter=True, name="conv_weight_2d")
        
        if self.with_bias:
            self.bias = Tensor.zeros(self.out_channel, requires_grad=True, 
                                    is_parameter=True, name="conv_bias_2d")
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, in_channel, width, height)
            Returns:
                Tensor: (batch_size, out_channel, output_dim1, output_dim2)
        """
        x_padded = x.pad2d(self.padding)
        if self.with_bias:
            return x_padded.conv2d(self.weight, self.stride) + self.bias.reshape(shape=[1, -1, 1, 1])
        return x_padded.conv2d(self.weight, self.stride)


class MaxPool1d(Module):
    """Performs a max pooling operation after a 1d convolution
        
        Args:
            kernel_size (int): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:int, stride:int=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x:Tensor) -> Tensor:
        """Performs a max pooling operation after a 1d convolution

        Args:
            x (Tensor): (batch_size, channel, in_width, in_height)

        Returns:
            Tensor: (batch_size, channel, out_width, out_height)
        """
        out = x.max_pool1d(self.kernel_size)
        out.name = 'mpool1d_res'
        return out


class AvgPool1d(Module):
    """Performs an average pooling operation after a 1d convolution
        
        Args:
            kernel_size (int): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:int, stride:int=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x:Tensor) -> Tensor:
        """Performs an average pooling operation after a 1d convolution

        Args:
            x (Tensor): (batch_size, channel, in_width, in_height)

        Returns:
            Tensor: (batch_size, channel, out_width, out_height)
        """
        out = x.avg_pool1d(self.kernel_size)
        out.name = 'avgpool1d_res'
        return out


class MaxPool2d(Module):
    """Performs a max pooling operation after a 2d convolution
    
        Args:
            kernel_size (tuple): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:tuple, stride:int=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, channel, in_width, in_height)
            
            Returns:
                Tensor: (batch_size, channel, out_width, out_height)
        """
        out = x.max_pool2d(self.kernel_size)
        out.name = 'mpool2d_res'
        return out


class AvgPool2d(Module):
    """Performs an average pooling operation after a 2d convolution

        Args:
            kernel_size (tuple): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:tuple, stride:int=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, channel, in_width, in_height)
            
            Returns:
                Tensor: (batch_size, channel, out_width, out_channel)
        """
        out = x.avg_pool2d(self.kernel_size)
        out.name = 'avgpool2d_res'
        return out


class Flatten(Module):
    """Layer that flattens all dimensions for each observation in a batch

        >>> x = torch.randn(4, 3, 2) # batch of 4 observations, each sized (3, 2)
        >>> x
        tensor([[[ 0.8816,  0.9773],
                [-0.1246, -0.1373],
                [-0.1889,  1.6222]],

                [[-0.5303,  0.3655],
                [-0.7496,  0.6935],
                [-0.8173,  0.4346]]])
        >>> layer = Flatten()
        >>> out = layer(x)
        >>> out
        tensor([[ 0.8816,  0.9773, -0.1246, -0.1373, -0.1889,  1.6222],
                [-0.5303,  0.3655, -0.7496,  0.6935, -0.8173,  0.4346]])
        >>> out.shape
        torch.size([4, 6]) # batch of 4 observations, each flattened into 1d array size (6,)

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        r"""
            Args:
                x (Tensor): (batch_size, dim_2, dim_3, ...) arbitrary number of dims after batch_size
            Returns:
                out (Tensor): (batch_size, dim_2 * dim_3 * ...) batch_size, then all other dims flattened
        """
        dim1, dim2 = x.shape[0], np.prod(x.shape[1:])
        out = x.reshape((dim1, dim2))
        out.name = 'flatten_res'
        return out


class Dropout(Module):
    """Dropout layer for regularization
    
        Args:
            p (float): Proportion of dropped out connections
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, p:float=0.5) -> None:
        super().__init__()
        self.mask, self.p = None, p
    
    def forward(self, x:Tensor) -> Tensor:
        """Forward pass

            In training mode: Remove connections with probability p
            In validation mode: Multiply by 1-p the output tensor

            Args:
                x (Tensor): Input tensor

            Returns:
                Tensor: Output tensor
        """
        if self.is_train:
            if self.mask is None:
                val = np.random.binomial(1, 1.0 - self.p, size=x.shape)
                self.mask = Tensor(val, name="dropout_mask", is_parameter=True)
            out = x * self.mask
            out.name = 'dropout_res'
            return out
        return x * Tensor(1 - self.p, name='dropout_prob', is_parameter=True)


class CrossEntropyLoss(Module):
    """CrossEntropyLoss layer

        During implementation of a model, you can either choose a CrossEntropyLoss layer
        or call the CrossEntropy function from nn.functional.
        
        >>> criterion = CrossEntropyLoss()
        >>> criterion(outputs, labels)
        3.241

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        pass

    def forward(self, predicted:Tensor, target:Tensor) -> Tensor:
        """Forward pass

           Args:
                predicted (Tensor): (batch_size, num_classes)
                target (Tensor): (batch_size,)
           Returns:
                Tensor: loss, stored as a float in a tensor 
        """
        return tensor.cross_entropy(predicted, target)
    

class MSELoss(Module):
    """The MSELoss function.
       Mean squared error function is used for regression problems.

        >>> criterion =  MSELoss()
        >>> criterion(outputs, labels)
        3.241

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        pass

    def forward(self, predicted:Tensor, target:Tensor) -> Tensor:
        """Forward pass
        
        Args:
            predicted (Tensor): (batch_size, num_classes)
            target (Tensor): (batch_size,)
        Returns:
            Tensor: loss, stored as a float in a tensor
        """
        return ((predicted - target) ** 2).mean()


# ***** Activation functions *****


class ReLU(Module):
    r"""
        ReLU Activation Layer

        Applies a Rectified Linear Unit activation function to 
        the input

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        """Forward pass.

            ReLU(x) = max(0, x)

            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x.relu()


class LeakyReLU(Module):
    r"""
        Leaky ReLU Activation Layer

        Applies a Leaky Rectified Linear Unit activation function to
        the input

        Args:
            alpha (float):

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, alpha=0.01) -> None:
        super().__init__()

        self.alpha = Tensor(alpha, is_parameter=False, name='alpha_relu')

    def forward(self, x:Tensor) -> Tensor:
        """Forward pass

            LeakyReLU(x) = max(alpha * x, x)

            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x.relu() - self.alpha * (-x).relu()
    

class Sigmoid(Module):
    r"""
        Sigmoid Activation Layer

        Applies a Sigmoid activation function to the input

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x.sigmoid()


class Tanh(Module):
    r"""
        Tanh Activation Layer

        Applies a Hyperbolic Tangent activation function to the input

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x.tanh()

class Swish(Module):
    r"""
        Swish Activation Layer

        Applies a Swish activation function to the input

        Beta is learnt by gradient descent

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, beta:float=1.0) -> None:
        super().__init__()

        self.beta = Tensor(beta, requires_grad=True, is_parameter=True, name='swish_beta')
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x * (self.beta * x).sigmoid()


class Mish(Module):
    r"""
        Mish Activation Layer

        Applies a Mish activation function to the input

        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x * (1 + x.exp()).log().tanh()