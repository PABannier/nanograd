from typing import Union
from collections import defaultdict
import warnings
import inspect
from functools import partialmethod
import functools

import numpy as np
from nanograd.viz.comp_graph import ForwardGraphVisualizer, BackwardGraphVisualizer

from nanograd.device import Device
from nanograd.nn.buffer import GPUBuffer

cl_ctx, cl_queue = None, None

def get_gpu_context_and_queue():
    """
        If GPU is enabled, get_gpu_context_and_queue populates global variables
        cl_ctx and cl_queue used for GPU computations. 
        
        cl_ctx and cl_queue are PyOpenCL objects used for parallelized computations 
        on the GPU. More precisely, cl_ctx is expected to be an instance of PyOpenCL 
        Context and cl_queue is expected to be an instance of PyOpenCL CommandQueue.
    """
    global cl_ctx, cl_queue
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
        devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)


class Tensor:
    ops = defaultdict(dict) # Adding to the class several operations
    """
        Tensor is the basic operator object of Nanograd. It is 
        a wrapper class around NumPy array. 

        Args:
            data (np.ndarray or int or float or GPUBuffer): Contains data to be stored. It can 
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

            device (Device, optional): the device to use for computations. Currently, Nanograd only
                supports CPU and GPU.

            name (str, optional): A name for the Tensor for visualization purposes.

            op (str, optional): The operation used to create the tensor. Could be an addition, a substraction,
                a product.

        ..note::
            If the node is gradient-enabled, the grad property is populated with a gradient Tensor
            object during backpropagation.
    """
    def __init__(self, 
                 data:Union[np.ndarray, GPUBuffer], 
                 requires_grad:bool=False, 
                 is_parameter:bool=False,
                 device:Device=Device.CPU,
                 name:str='no_name',
                 op:str=None) -> None:
        self.data = self._move_data(data, device)
        self.requires_grad = requires_grad
        self.is_parameter = is_parameter

        self.device = device

        self.grad = None
        self.ctx = None

        self.name, self.op = name, op
        self.children = []
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()
    
    # ****************************************
    # ***** Class methods / Initializers *****
    # ****************************************

    @classmethod
    def zeros(cls, *shape, **kwargs):
        """Creates a Tensor filled with zeros"""
        return cls(np.zeros(*shape, dtype=np.float32), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        """Creates a Tensor filled with ones"""
        return cls(np.ones(*shape, dtype=np.float32), **kwargs)
    
    @classmethod
    def arange(cls, *interval, **kwargs):
        """Creates a Tensor filled with values in range"""
        return cls(np.arange(*interval).astype(np.float32), **kwargs)
    
    @classmethod
    def randn(cls, *shape, **kwargs):
        """
            Creates a Tensor filled with values drawn from the
            standard Gaussian distribution
        """
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)
    
    @classmethod
    def normal(cls, loc, scale, *shape, **kwargs):
        """
            Creates a Tensor filled with values drawn from the
            a custom Gaussian distribution
        """
        return cls(np.random.normal(loc, scale, *shape).astype(np.float32), **kwargs)

    @classmethod
    def randint(cls, low, high, *shape, **kwargs):
        """
            Creates a Tensor filled with integer values drawn in the
            range between low and high
        """
        return cls(np.random.randint(low, high, *shape).astype(np.float32), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        """
            Creates a 2-dimensional Tensor (matrix) equal to the
            identity matrix.
        """
        return cls(np.eye(dim).astype(np.float32), **kwargs)
    
    # ****************************************
    # *********** CPU/GPU support ************
    # ****************************************

    @staticmethod
    def _move_data(data:Union[np.ndarray, GPUBuffer], device:Device):
        """
            Moves data to the device specified

            Args:
                data (np.ndarray or GPUBuffer): data to be moved
                device (Device): the destination device
            
            Returns:
                data (np.ndarray or GPUBuffer): numpy array if CPU else cl.Buffer
        """
        assert device in Device, "Unsupported device. Only CPU and GPU available."

        if isinstance(data, GPUBuffer):
            if device == Device.GPU:
                return data
            cpu_data = np.empty(data.shape, dtype=np.float32)
            cl.enqueue_copy(cl_queue, cpu_data, data.cl, is_blocking=True)
            return cpu_data
        
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if device == Device.GPU:
            if cl_ctx is None:
                get_gpu_context_and_queue()
            return GPUBuffer(cl_ctx, data.shape, hostbuf=data)

        return data

    def to(self, device:Device):
        """
            Moves the Tensor to the specified device 
            
            Args:
                device (Device): the destination device
        """
        self.data, self.device = self._move_data(self.data, device), device
        if self.grad:
            self.grad.to(device) # Recursive call to move the gradient to specified device

    def cpu(self):
        self.to(Device.CPU)
        return self
    
    def gpu(self):
        if not PYOPENCL_AVAILABLE:
            raise Exception("OpenCL is not installed in this environment. Please consider running \
                             pip install pyopencl to benefit from GPU-accelerated computations.")
        self.to(Device.GPU)
        return self

    # ****************************************
    # *************** Backprop ***************
    # ****************************************

    @functools.lru_cache()
    def build_graph_topology(self):
        def dfs(node, visited, nodes):
            visited.add(node)
            if node.ctx:
                for parent in node.ctx.parents:
                    if parent not in visited:
                        dfs(parent, visited, nodes)
                nodes.append(node)
            return nodes
        return dfs(self, set(), list())
        
    def backward(self):
        if self.shape != (1,):
            raise Exception("Can't initiate backprop from a non scalar-valued tensor.")

        self.grad = Tensor.ones(self.shape, device=self.device, requires_grad=False)

        for node in reversed(self.build_graph_topology()):
            assert node.grad is not None, 'Got an unitialized gradient node'

            gradients = node.ctx.backward(node.ctx, node.grad.data)

            if len(node.ctx.parents) == 1:
                gradients = [gradients]
            
            for tensor, grad in zip(node.ctx.parents, gradients):
                if grad is not None: 
                    assert grad.shape == tensor.shape, f"Mismatched tensor and grad shape. Got {grad.shape} and {tensor.shape}. \
                                                         Tensor and gradient should have the same shape."
                    if tensor.grad is None:
                        tensor.grad = Tensor(grad, device=self.device, requires_grad=False)
                    else:
                        tensor.grad += Tensor(grad, device=self.device, requires_grad=False)

    def copy(self):
        """Creates a copy of the tensor"""
        return Tensor(self.data, device=self.device)

    def __str__(self):
        return f"<NanoTensor({str(self.data)}, " + \
               f"name={self.name}," + \
               f"device={self.device}>"
    
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
            elif not np.issubdtype(type(start), int):
                raise TypeError(f"Indices must be integer. Got {type(start)}")
            
            if stop is None:
                stop = self.shape[i]
            elif not np.issubdtype(type(stop), int):
                raise TypeError(f"Indices must be integer. Got {type(stop)}")
            elif stop < 0:
                stop = self.shape[i] + stop

            assert arg.step is None or arg.step == 1, "Custom step not yet implemented"
            indices.append((start, stop))
        
        indices += [(0, shape[i]) for i in range(len(args), len(self.shape))]
        
        return self.slice(indices=indices)

    # ****************************************
    # ********** Basic operations ************
    # ****************************************
    
    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def sqrt(self):
        return self ** 0.5
    
    # ****************************************
    # ******** Reduction operations **********
    # ****************************************
    
    def mean(self, axis=None):
        out = self.sum(axis=axis)
        coeff = np.prod(out.shape) / np.prod(self.shape)
        return out * coeff

    # ****************************************
    # ****** Miscellaneous operations ********
    # ****************************************

    def flatten(self):
        dim1, dim2 = self.shape[0], np.prod(self.shape[1:])
        out = self.reshape(shape=(dim1, dim2))
        return out

    # ****************************************
    # ******** Activation functions **********
    # ****************************************
    
    def log_softmax(self):
        batch_size, num_classes = self.shape
        a = self.max(axis=1).reshape(shape=[batch_size, 1]) # Log-exp trick
        out = self - a - (self - a).exp().sum(axis=1).log().reshape(shape=[batch_size, 1])
        return out

    # ****************************************
    # ********* Pool operations *********
    # ****************************************

    def _pool1d(self, field_length:int):
        """1-dimensional pooling operation

        Args:
            field_length (int): length of the pooling kernel
        
        Returns:
            x_reshaped (Tensor): pooled Tensor
        """
        x_unpadded = self[:, :, :self.shape[2] - self.shape[2] % field_length]

        shape = (x_unpadded.shape[0], x_unpadded.shape[1], 
                 x_unpadded.shape[2] // field_length, field_length)

        x_reshaped = x_unpadded.reshape(shape=shape)
        return x_reshaped 
    
    def max_pool1d(self, kernel_size:int=2):
        """MaxPooling1d operation

        Args:
            kernel_size (int, optional): Kernel length for pooling operation. Defaults to 2.
        """
        return self._pool1d(kernel_size).max(axis=3)
    
    def avg_pool1d(self, kernel_size:int=2):
        """AvgPooling1d operation

        Args:
            kernel_size (int, optional): Kernel length for pooling operation. Defaults to 2.
        """
        return self._pool1d(kernel_size).mean(axis=3)
        
    def _pool2d(self, pool_size:tuple):
        """
            2-dimensional pooling operation

            Args:
                pool_size (tuple): height of the pooling kernel
            
            Returns:
                x_reshaped (Tensor): pooled Tensor
        """
        x_unpadded = self[:, :, 
            :self.shape[2] - self.shape[2] % pool_size[0], 
            :self.shape[3] - self.shape[3] % pool_size[1]]
        
        shape = (x_unpadded.shape[0], x_unpadded.shape[1], 
                 x_unpadded.shape[2] // pool_size[0], pool_size[0], 
                 x_unpadded.shape[3] // pool_size[1], pool_size[1])

        x_reshaped = x_unpadded.reshape(shape=shape)

        return x_reshaped
    
    def max_pool2d(self, pool_size:tuple=(2, 2)):
        """MaxPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return self._pool2d(pool_size).max(axis=(3, 5))
    
    def avg_pool2d(self, pool_size:tuple=(2, 2)):
        """AvgPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return self._pool2d(pool_size).mean(axis=(3, 5))

    def pad1d(self, pad:tuple):
        """Padding for one-dimensional signal

        Args:
            pad (tuple): Amount of padding before and after the signal

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[0]:int(self.shape[2])+pad[1]]

    def pad2d(self, pad:tuple):
        """Padding for two-dimensional images

        Args:
            pad (tuple): 4-dimensional tuple. Amount of padding to be applied before and
                         after the signal along 2 dimensions

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[2]:int(self.shape[2])+pad[3], -pad[0]:int(self.shape[3])+pad[1]]


    # ****************************************
    # ************ Visualization *************
    # ****************************************

    def plot_forward(self, rankdir="LR"):
        r"""
            Plots a forward computational graph

            Args:
                rankdir (str): LR (left to right) and TB (top to bottom)
        """
        visualizer = ForwardGraphVisualizer()
        return visualizer.visualize(self, rankdir=rankdir)
    
    def plot_backward(self, rankdir="LR"):
        r"""
            Plots a backward computational graph

            Args:
                rankdir (str): LR (left to right) and TB (top to bottom)
        """
        visualizer = BackwardGraphVisualizer()
        return visualizer.visualize(self, rankdir=rankdir)


def register(name, operation, device=Device.CPU):
    """Registers operation to the Tensor class

       More precisely, it populates the ops dictionary by adding CPU and 
       GPU operations.

       In short:
            ops[device][name] = function

    Args:
        name (str): Key representing the function stored in the ops dictionary
        function (class, inherits from Function): Value in the dictionary
        device ([type], optional): The device on which the operation is execute. Defaults to Device.CPU.
    """

    Tensor.ops[device][name] = operation
    
    def dispatch(*args, **kwargs):
        """Modifies the operation arguments and adds the operation 
           to the ops dictionary

        Returns:
            function: Forward pass (apply) of the operation
        """
        input = [arg for arg in args if isinstance(arg, Tensor)][0]
        args = [Tensor(np.array([arg], dtype=input.dtype), device=input.device) 
                if not isinstance(arg, Tensor) else arg for arg in args]
        
        op = Tensor.ops[input.device][name]
        op.cl_ctx, op.cl_queue, op.device = cl_ctx, cl_queue, input.device # For GPU support
        return op.apply(op, *args, **kwargs)

    if name in ['add', 'mul', 'pow', 'matmul', 'neg']:
        setattr(Tensor, f"__{name}__", dispatch)
        if name != 'neg':
            setattr(Tensor, f"__r{name}__", lambda self, x: dispatch(x, self))
    else:
        setattr(Tensor, name, dispatch)


def register_ops(namespace, device=Device.CPU):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_":  
            register(name.lower(), cls, device=device)


from nanograd.nn import ops_cpu

register_ops(ops_cpu)

try:
    import pyopencl as cl
    from nanograd.nn import ops_gpu
    register_ops(ops_gpu, device=Device.GPU)

    PYOPENCL_AVAILABLE = True

except ImportError:
    PYOPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL is not available on this computer. Can't use \
                   parallel computing. Please install it to move comptutations \
                   to move the GPU.")
