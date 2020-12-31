import numpy as np
import autograd_engine
from nn import functional as F

class Tensor:
    r"""
        Tensor is the basic operator object of Nanograd. It is 
        a wrapper class around NumPy array. 

        Arguments:
            data (np.array or int or float): Contains data to be stored. It can 
                be a scalar, a vector, a matrix or a tensor (multi-dimensional array).

            requires_grad (bool, optional): If ``True``, a gradient will be stored and 
                accumulated as a property of an object. If ``False``, the grad property 
                remains None. No gradient is stored in the object.

            is_leaf (bool, optional): If ``True``, the corresponding node in the computational
                graph does not have any parents. Note that it is usually the case for parameters
                in a neural network. If ``False``, the node is expected to have parents. 
                Usually a non-leaf node results from a basic operation.

            is_parameter (bool, optional): the Tensor contains trainable parameters. It is useful
                when building a neural network.

        ..note::

            If the node is gradient-enabled, the grad property is populated with a gradient Tensor
            object during backpropagation.
    """
    def __init__(self, 
                 data, 
                 requires_grad=False, 
                 is_leaf=True, 
                 is_parameter=False):
        
        self.data = np.array(data)

        self.is_parameter = is_parameter
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.is_parameter = is_parameter
        
        self.grad = None 
        self.grad_fn = None

        if not self.requires_grad and not self.is_leaf:
            raise Exception("A non-leaf node must be gradient-enabled")

    def backward(self):
        r"""Initiates the gradient backpropagation"""
        autograd_engine.backward(self.grad_fn, Tensor.ones(self.shape))
    
    def copy(self):
        return Tensor(self.data)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __str__(self):
        return f"NanoTensor({str(self.data)}, " + \
               f"grad_fn={self.grad_fn.__class__.__name__  if self.grad_fn else None})"
    
    def __repr__(self):
        return self.__str__()

    def __getitem__(self, args):
        if args is None: 
            args = []
        elif type(args) in [list, tuple]: 
            pass
        else: 
            args = [args]

        indices = []

        for i, arg in enumerate(args):
            start, stop = arg.start, arg.stop

            if start is None:
                start = 0
            elif type(start) != int:
                raise TypeError(f"Indices must be integer. Got {type(start)}")
            
            if stop is None:
                stop = self.shape[i]
            elif type(stop) != int:
                raise TypeError(f"Indices must be integer. Got {type(stop)}")
            elif stop < 0:
                stop = self.shape[i] + stop

            assert arg.step is None or arg.step == 1, "Custom step not yet implemented"
            indices.append((start, stop))
        
        indices += [(0, shape[i]) for i in range(len(args), len(self.shape))]
        
        return F.Slice.apply(self, indices)
        
    def __add__(self, other):
        return F.Add.apply(self, other)
    
    def __sub__(self, other):
        return F.Sub.apply(self, other)
    
    def __neg__(self):
        return F.Neg.apply(self)

    def __pow__(self, exp):
        return F.Pow.apply(self, exp)
    
    def __mul__(self, other):
        return F.Mul.apply(self, other)
    
    def __matmul__(self, other):
        return F.MatMul.apply(self, other)
    
    def __truediv__(self, other):
        return F.Div.apply(self, other)
    
    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)
    
    def mean(self, axis=None, keepdims=False):
        out = self.sum(axis=axis)
        coeff = np.prod(out.shape) / np.prod(self.shape)
        return out * Tensor(coeff)

    def max(self, axis=None):
        return F.Max.apply(self, axis)
    
    def log(self):
        return F.Log.apply(self)
    
    def exp(self):
        return F.Exp.apply(self)

    def sqrt(self):
        return F.Sqrt.apply(self)
    
    def reshape(self, shape):
        return F.Reshape.apply(self, shape)
    
    def T(self):
        return F.Transpose.apply(self)
    
    def relu(self):
        return F.ReLU.apply(self)
    
    def sigmoid(self):
        return F.Sigmoid.apply(self)

    def tanh(self):
        return F.Tanh.apply(self)
    
    def _pool2d(self, field_height, field_width):
        x_unpadded = self[:, :, 
                          :self.shape[2] - self.shape[2] % field_height, 
                          :self.shape[3] - self.shape[3] % field_width]
        
        x_reshaped = x_unpadded.reshape(
            shape=(x_unpadded.shape[0], x_unpadded.shape[1], 
                   x_unpadded.shape[2] // field_height, field_height, 
                   x_unpadded.shape[3] // field_width, field_width)
        )

        return x_reshaped
    
    def max_pool2d(self, kernel_size=(2, 2)):
        return self._pool2d(*kernel_size).max(axis=5).max(axis=3)
    
    def avg_pool2d(self, kernel_size=(2, 2)):
        return self._pool2d(*kernel_size).mean(axis=(3, 5))
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(*shape), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(*shape), **kwargs)
    
    @classmethod
    def arange(cls, *interval, **kwargs):
        return cls(np.arange(*interval), **kwargs)
    
    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape), **kwargs)
    
    @classmethod
    def normal(cls, loc, scale, *shape, **kwargs):
        return cls(np.random.normal(loc, scale, *shape), **kwargs)

    @classmethod
    def randint(cls, low, high, *shape, **kwargs):
        return cls(np.random.randint(low, high, *shape), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim), **kwargs)