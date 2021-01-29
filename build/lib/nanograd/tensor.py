from typing import Union
import warnings

import numpy as np
from nanograd.viz.comp_graph import ForwardGraphVisualizer, BackwardGraphVisualizer

from nanograd.device import Device
from nanograd import autograd_engine
from nanograd.nn import functional as F
from nanograd.nn.ops_gpu import GPUBuffer


try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True

except ImportError:
    PYOPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL is not available on this computer. Can't use \
                   parallel computing. Please install it to move comptutations \
                   to move the GPU.")


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
                 is_leaf:bool=True, 
                 is_parameter:bool=False,
                 device:Device=Device.CPU,
                 name:str='no_name',
                 op:str=None) -> None:
        self.data = self._move_data(data, device)
        self.requires_grad, self.is_leaf = requires_grad, is_leaf
        self.is_parameter = is_parameter

        self.device = device

        self.grad, self.grad_fn = None, None
        self.name, self.op = name, op
        self.children = []

        if not self.requires_grad and not self.is_leaf:
            raise Exception("A non-leaf node must be gradient-enabled")
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    # ****************************************
    # ***** Class methods / Initializers *****
    # ****************************************

    @classmethod
    def zeros(cls, *shape, **kwargs):
        """Creates a Tensor filled with zeros"""
        return cls(np.zeros(*shape), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        """Creates a Tensor filled with ones"""
        return cls(np.ones(*shape), **kwargs)
    
    @classmethod
    def arange(cls, *interval, **kwargs):
        """Creates a Tensor filled with values in range"""
        return cls(np.arange(*interval), **kwargs)
    
    @classmethod
    def randn(cls, *shape, **kwargs):
        """
            Creates a Tensor filled with values drawn from the
            standard Gaussian distribution
        """
        return cls(np.random.randn(*shape), **kwargs)
    
    @classmethod
    def normal(cls, loc, scale, *shape, **kwargs):
        """
            Creates a Tensor filled with values drawn from the
            a custom Gaussian distribution
        """
        return cls(np.random.normal(loc, scale, *shape), **kwargs)

    @classmethod
    def randint(cls, low, high, *shape, **kwargs):
        """
            Creates a Tensor filled with integer values drawn in the
            range between low and high
        """
        return cls(np.random.randint(low, high, *shape), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        """
            Creates a 2-dimensional Tensor (matrix) equal to the
            identity matrix.
        """
        return cls(np.eye(dim), **kwargs)
    
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
            data = np.array([data]).astype(np.float32)

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
    
    def gpu(self):
        if not PYOPENCL_AVAILABLE:
            raise Exception("OpenCL is not installed in this environment. Please consider running \
                             pip install pyopencl to benefit from GPU-accelerated computations.")
        self.to(Device.GPU)
        

    # ****************************************
    # *************** Backprop ***************
    # ****************************************

    def backward(self) -> None:
        """
            Initiates backpropagation.

            ..note::
                The gradient of the loss with respect to the parameters is a multi-dimensional array filled
                with ones.
        """
        if not self.requires_grad:
            raise Exception("Can't initiate backprop from gradient-disabled tensor")
        autograd_engine.backward(self.grad_fn, Tensor.ones(self.shape, device=self.device))
    
    def copy(self):
        """Creates a copy of the tensor"""
        return Tensor(self.data, device=self.device)

    # ****************************************
    # *********** Magic functions ************
    # ****************************************
    
    def __str__(self):
        return f"<NanoTensor({str(self.data)}, " + \
               f"grad_fn={self.grad_fn.__class__.__name__  if self.grad_fn else None}), " + \
               f"name={self.name}>"
    
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
        
        return F.Slice.apply(self, indices, cl_ctx=cl_ctx, cl_queue=cl_queue)

    # ****************************************
    # ********** Basic operations ************
    # ****************************************
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return F.Add.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return F.Neg.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __sub__(self, other):
        return self + (- other)
    
    def __rsub__(self, other):
        return -self + other

    def __pow__(self, exp):
        return F.Pow.apply(self, exp, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return F.Mul.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** (-1.0))
    
    def __matmul__(self, other):
        return F.MatMul.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    # ****************************************
    # ******** Advanced operations ***********
    # ****************************************
    
    def sum(self, axis=None, keepdims:bool=False):
        return F.Sum.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def mean(self, axis=None, keepdims:bool=False):
        out = self.sum(axis=axis)
        coeff = np.prod(out.shape) / np.prod(self.shape)
        return out * Tensor(coeff, device=self.device)
    
    def reshape(self, shape:tuple):
        return F.Reshape.apply(self, shape, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def T(self):
        return F.Transpose.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def max(self, axis=None, keepdims=False):
        return F.Max.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def min(self, axis=None, keepdims=False):
        return F.Min.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def log(self):
        return F.Log.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def exp(self):
        return F.Exp.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def sqrt(self):
        return self ** (1/2)

    # ****************************************
    # ****** Miscellaneous operations ********
    # ****************************************
    
    def one_hot(self, num_classes):
        return F.OneHot.apply(self, num_classes, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def unsqueeze(self, axis):
        return F.Unsqueeze.apply(self, axis, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def squeeze(self, axis):
        return F.Squeeze.apply(self, axis, cl_ctx=cl_ctx, cl_queue=cl_queue)

    # ****************************************
    # ******** Activation functions **********
    # ****************************************

    def relu(self):
        return F.ReLU.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def sigmoid(self):
        return F.Sigmoid.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def tanh(self):
        return F.Tanh.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    # ****************************************
    # ********* Conv/Pool operations *********
    # ****************************************
    
    def _pool2d(self, field_height:int, field_width:int):
        r"""
            2-dimensional pooling operation

            Args:
                field_height (int): height of the pooling kernel
                field_width (int): width of the pooling kernel
            
            Returns:
                x_reshaped (Tensor): pooled Tensor
        """
        x_unpadded = self[:, :, 
            :self.shape[2] - self.shape[2] % field_height, 
            :self.shape[3] - self.shape[3] % field_width]
        
        shape = (x_unpadded.shape[0], x_unpadded.shape[1], 
                 x_unpadded.shape[2] // field_height, field_height, 
                 x_unpadded.shape[3] // field_width, field_width)

        x_reshaped = x_unpadded.reshape(shape=shape)

        return x_reshaped
    
    def max_pool2d(self, kernel_size:tuple=(2, 2)):
        """MaxPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return self._pool2d(*kernel_size).max(axis=5).max(axis=3)
    
    def avg_pool2d(self, kernel_size:tuple=(2, 2)):
        """AvgPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return self._pool2d(*kernel_size).mean(axis=(3, 5))

    def pad1d(self, pad:tuple):
        """Padding for one-dimensional signal

        Args:
            pad (tuple): Amount of padding before and after the signal

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[0]:self.shape[2]+pad[1]]

    def pad2d(self, pad:tuple):
        """Padding for two-dimensional images

        Args:
            pad (tuple): 4-dimensional tuple. Amount of padding to be applied before and
                         after the signal along 2 dimensions

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[2]:self.shape[2]+pad[3], -pad[0]:self.shape[3]+pad[1]]

    def conv1d(self, weight, stride:int):
        """1d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, kernel_length)
            stride (int): Stride of the convolution operation
        """
        return F.Conv1d.apply(self, weight, stride, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def conv2d(self, weight, stride):
        """2d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, *kernel_size)
            stride (int): Stride of the convolution operation
        """
        return F.Conv2d.apply(self, weight, stride, cl_ctx=cl_ctx, cl_queue=cl_queue)

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