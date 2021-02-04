from typing import Union
import warnings

import numpy as np
from nanograd.viz.comp_graph import ForwardGraphVisualizer, BackwardGraphVisualizer

from nanograd.device import Device
from nanograd import autograd_engine
from nanograd.autograd_engine import Function
from nanograd.nn.ops_gpu import GPUBuffer

import nanograd.nn.ops_cpu as ops_cpu
import nanograd.nn.ops_gpu as ops_gpu
from nanograd.nn.conv_ops import (get_conv1d_output_size, get_conv2d_output_size, 
                                  get_im2col_indices, col2im)

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
        
        return Slice.apply(self, indices, cl_ctx=cl_ctx, cl_queue=cl_queue)

    # ****************************************
    # ********** Basic operations ************
    # ****************************************
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return Add.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return Neg.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __sub__(self, other):
        return self + (- other)
    
    def __rsub__(self, other):
        return -self + other

    def __pow__(self, exp):
        return Pow.apply(self, exp, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return Mul.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1.0)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    # ****************************************
    # ******** Advanced operations ***********
    # ****************************************
    
    def sum(self, axis=None, keepdims:bool=False):
        return Sum.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def mean(self, axis=None, keepdims:bool=False):
        out = self.sum(axis=axis)
        coeff = np.prod(out.shape) / np.prod(self.shape)
        return out * Tensor(coeff, device=self.device)
    
    def reshape(self, shape:tuple):
        return Reshape.apply(self, shape, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def T(self):
        return Transpose.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def max(self, axis=None, keepdims=False):
        return Max.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def min(self, axis=None, keepdims=False):
        return Min.apply(self, axis, keepdims, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def log(self):
        return Log.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def exp(self):
        return Exp.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def sqrt(self):
        return self ** (1/2)

    # ****************************************
    # ****** Miscellaneous operations ********
    # ****************************************
    
    def one_hot(self, num_classes):
        return OneHot.apply(self, num_classes, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def unsqueeze(self, axis):
        return Unsqueeze.apply(self, axis, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def squeeze(self, axis):
        return Squeeze.apply(self, axis, cl_ctx=cl_ctx, cl_queue=cl_queue)

    # ****************************************
    # ******** Activation functions **********
    # ****************************************

    def relu(self):
        return ReLU.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def sigmoid(self):
        return Sigmoid.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)

    def tanh(self):
        return Tanh.apply(self, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    # ****************************************
    # ********* Conv/Pool operations *********
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
        
    def _pool2d(self, field_height:int, field_width:int):
        """
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

    def conv1d(self, weight, stride:int):
        """1d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, kernel_length)
            stride (int): Stride of the convolution operation
        """
        return Conv1d.apply(self, weight, stride, cl_ctx=cl_ctx, cl_queue=cl_queue)
    
    def conv2d(self, weight, stride):
        """2d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, *kernel_size)
            stride (int): Stride of the convolution operation
        """
        return Conv2d.apply(self, weight, stride, cl_ctx=cl_ctx, cl_queue=cl_queue)

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


# ****************************************
# ************** Functional **************
# ****************************************

def cross_entropy(predicted:Tensor, target:Tensor) -> Tensor:
    """Calculates Cross Entropy Loss between logits and true labels.
       Used in the CrossEntropy module

    Args:
        predicted (Tensor): Logits
        target (Tensor): Target classes

    Returns:
        Tensor: Loss in a Tensor of shape ()
    """
    batch_size, num_classes = predicted.shape
    labels = target.one_hot(num_classes)

    a = predicted.max(axis=1).reshape((batch_size, 1))
    log_softmax = predicted - a - (predicted - a).exp().sum(axis=1).log().reshape((batch_size, 1))
    nll_loss = - (log_softmax * labels).sum() / batch_size

    return nll_loss


class OneHot(Function):
    @staticmethod
    def forward(ctx, a, num_classes):
        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        if a.device == Device.CPU:
            out = ops_cpu.one_hot_encoding(a.data, num_classes)
        else:
            out = ops_gpu.one_hot_encoding(ctx.cl_ctx, ctx.cl_queue, a.data, num_classes)
        
        return Tensor(out, device=a.device, requires_grad=requires_grad,
                      is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        # No backward pass since one-hot encoding is applied to a target
        # tensor whose gradient is None
        return None


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, axis):
        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        ctx.axis = axis

        if a.device == Device.CPU:
            out = ops_cpu.unsqueeze_forward(a.data, axis)
        else:
            out = ops_gpu.unsqueeze_forward(ctx.cl_ctx, ctx.cl_queue, a, axis)

        return Tensor(out, device=a.device, requires_grad=requires_grad,
                      is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis

        if grad_output.device == Device.CPU:
            grad = ops_cpu.unsqueeze_backward(grad_output.data, axis)
        else:
            grad = ops_gpu.unsqueeze_backward(ctx.cl_ctx, ctx.cl_queue, grad_output, axis)
        
        return Tensor(grad, device=grad_output.device), None


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, axis):
        is_squeezed = False
        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        if a.shape[axis] == 1: # If dimension > 1, can't squeeze
            is_squeezed = True
            if a.device == Device.CPU:
                out = ops_cpu.squeeze_forward(a.data, axis)
            else:
                out = ops_gpu.squeeze_forward(ctx.cl_ctx, ctx.cl_queue, a, axis)
        else:
            is_squeezed = False
            out = a.data

        ctx.axis = axis
        ctx.is_squeezed = is_squeezed
        
        return Tensor(out, device=a.device, requires_grad=requires_grad,
                      is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis, is_squeezed = ctx.axis, ctx.is_squeezed

        if is_squeezed:
            if grad_output.device == Device.CPU:
                grad = ops_cpu.squeeze_backward(grad_output.data, axis)
            else:
                grad = ops_gpu.squeeze_backward(ctx.cl_ctx, ctx.cl_queue, grad_output, axis)
        else:
            grad = grad_output.data
        
        return Tensor(grad, device=grad_output.device), None


class Slice(Function):
    @staticmethod
    def forward(ctx, a, indices=None):
        ctx.shape, ctx.indices = a.shape, indices

        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.slice_forward(a.data, indices)
        else:
            out_data = ops_gpu.slice_forward(ctx.cl_ctx, ctx.cl_queue, a.data, indices)
        
        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'slice'

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        shape, fwd_indices = ctx.shape, ctx.indices

        if grad_output.device == Device.CPU:
            grad = ops_cpu.slice_backward(grad_output.data, shape, fwd_indices)
        else:
            grad = ops_gpu.slice_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, shape, fwd_indices)
        
        return Tensor(grad, device=grad_output.device), None


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if len(a.shape) > 2:
            raise Exception("Arg for Transpose must be 1D or 2D tensor: {}".format(a.shape))
        
        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.transpose_forward(a.data)
        else:
            out_data = ops_gpu.transpose_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                     is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'transpose'
        return out

    @staticmethod
    def backward(ctx, grad_output):

        if grad_output.device == Device.CPU:
            grad = ops_cpu.transpose_backward(grad_output.data)
        else:
            grad = ops_gpu.transpose_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data)

        return Tensor(grad, device=grad_output.device), None


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.reshape_forward(a.data, shape)
        else:
            out_data = ops_gpu.reshape_forward(ctx.cl_ctx, ctx.cl_queue, a, shape)
        
        out = Tensor(out_data, requires_grad=requires_grad, 
                     is_leaf=is_leaf, device=a.device)
        
        assert np.prod(out.shape) == np.prod(a.shape), "Inconsistent array reshape size"

        out.children = [a]
        out.op = 'reshape'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.device == Device.CPU:
            grad = ops_cpu.reshape_backward(grad_output.data, ctx.shape)
        else:
            grad = ops_gpu.reshape_backward(ctx.cl_ctx, ctx.cl_queue, 
                                            grad_output.data, ctx.shape)

        return Tensor(grad, device=grad_output.device), None


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        axis = [axis] if type(axis) == int else axis

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.max_forward(a.data, axis, keepdims)
        else:
            out_data = ops_gpu.max_forward(ctx.cl_ctx, ctx.cl_queue, 
                                           a.data, axis, keepdims)
        
        ctx.axis, ctx.out = axis, out_data

        out = Tensor(out_data, requires_grad=requires_grad, 
                     is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'max'

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        axis, out = ctx.axis, ctx.out
        inp = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad = ops_cpu.max_backward(grad_output, inp.data, out, axis)
        else:
            grad = ops_gpu.max_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                        inp.data, out, axis)
            
        return Tensor(grad, device=grad_output.device), None


class Min(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        axis = [axis] if type(axis) == int else axis

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.min_forward(a.data, axis, keepdims)
        else:
            out_data = ops_gpu.min_forward(ctx.cl_ctx, ctx.cl_queue, a.data, axis, keepdims)
        
        ctx.axis, ctx.out = axis, out_data

        out = Tensor(out_data, requires_grad=requires_grad, 
                     is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'min'

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        axis, out = ctx.axis, ctx.out
        inp = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad = ops_cpu.min_backward(grad_output, inp.data, out, axis)
        else:
            grad = ops_gpu.min_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                        inp.data, out, axis)

        return Tensor(grad, device=grad_output.device), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.log_forward(a.data)
        else:
            out_data = ops_gpu.log_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=not requires_grad, device=a.device)
        out.children = [a]   
        out.op = 'log'                
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.log_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.log_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return Tensor(grad_a, device=grad_output.device), None
    

class Exp(Function):
    @staticmethod
    def forward(ctx, a):   
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.exp_forward(a.data)
        else:
            out_data = ops_gpu.exp_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=not requires_grad, device=a.device)
        out.children = [a]
        out.op = 'exp'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.exp_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.exp_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return Tensor(grad_a, device=grad_output.device), None


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.device != b.device:
            if a.device == Device.CPU and b.device == Device.GPU:
                a.gpu()
            elif a.device == Device.GPU and b.device == Device.CPU:
                b.gpu()

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.add_forward(a.data, b.data)
        else:
            out_data = ops_gpu.add_forward(ctx.cl_ctx, ctx.cl_queue, a.data, b.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=not requires_grad, device=a.device)
        out.children = [a, b]
        out.op = 'add'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        if grad_output.device == Device.CPU:
            grad_a, grad_b = ops_cpu.add_backward(grad_output.data, a.shape, b.shape)
        else:
            grad_a, grad_b = ops_gpu.add_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data,
                                                  a.shape, b.shape)
        return Tensor(grad_a, device=grad_output.device), Tensor(grad_b, device=grad_output.device)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.axis = axis
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.sum_forward(a.data, axis, keepdims)
        else:
            out_data = ops_gpu.sum_forward(ctx.cl_ctx, ctx.cl_queue, a.data, axis, keepdims)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'sum'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad = ops_cpu.sum_backward(grad_output.data, a.data, axis)
        else:
            grad = ops_gpu.sum_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data, axis)
        
        return Tensor(grad, device=grad_output.device), None, None


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.neg_forward(a.data)
        else:
            out_data = ops_gpu.neg_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=not a.requires_grad, device=a.device)
        out.children = [a]
        out.op = 'neg'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.device == Device.CPU:
            grad = ops_cpu.neg_backward(grad_output.data)
        else:
            grad = ops_gpu.neg_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data)
        return Tensor(grad, device=grad_output.device), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.shape[1] != b.shape[0]:
            raise Exception(f"Shapes don't match: {a.shape}, {b.shape}")

        if a.device != b.device:
            if a.device == Device.CPU and b.device == Device.GPU:
                a.gpu()
            elif a.device == Device.GPU and b.device == Device.CPU:
                b.gpu()

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.matmul_forward(a.data, b.data)
        else:
            out_data = ops_gpu.matmul_forward(ctx.cl_ctx, ctx.cl_queue, a.data, b.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
        out.children = [a, b]
        out.op = 'matmul'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        if grad_output.device == Device.CPU:
            grad_a, grad_b = ops_cpu.matmul_backward(grad_output.data, a.data, b.data)
        else:
            grad_a, grad_b = ops_gpu.matmul_backward(ctx.cl_ctx, ctx.cl_queue, 
                                                     grad_output.data, a.data, b.data)
        return Tensor(grad_a, device=grad_output.device), Tensor(grad_b, device=grad_output.device)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exp):
        if not isinstance(exp, (int, float)):
            raise Exception("Power can only be float or int")

        ctx.save_for_backward(a)
        ctx.exp = exp

        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.pow_forward(a.data, exp)
        else:
            out_data = ops_gpu.pow_forward(ctx.cl_ctx, ctx.cl_queue, a.data, exp)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=not requires_grad, device=a.device)
        out.children = [a]
        out.op = 'pow'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        exp = ctx.exp
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.pow_backward(grad_output.data, a.data, exp)
        else:
            grad_a = ops_gpu.pow_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data, exp)

        return Tensor(grad_a, device=grad_output.device), None


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.device != b.device:
            if a.device == Device.CPU and b.device == Device.GPU:
                a.gpu()
            elif a.device == Device.GPU and b.device == Device.CPU:
                b.gpu()

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.mul_forward(a.data, b.data)
        else:
            out_data = ops_gpu.mul_forward(ctx.cl_ctx, ctx.cl_queue, a.data, b.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                     is_leaf=not requires_grad, device=a.device)
        out.children = [a, b]
        out.op = 'mul'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        if grad_output.device == Device.CPU:
            grad_a, grad_b = ops_cpu.mul_backward(grad_output.data, a.data, b.data)
        else:
            grad_a, grad_b = ops_gpu.mul_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data, b.data)
        
        return Tensor(grad_a, device=grad_output.device), Tensor(grad_b, device=grad_output.device)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.relu_forward(a.data)
        else:
            out_data = ops_gpu.relu_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'relu'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.relu_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.relu_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return Tensor(grad_a, device=grad_output.device), None


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.sigmoid_forward(a.data)
        else:
            out_data = ops_gpu.sigmoid_forward(ctx.cl_ctx, ctx.cl_queue, a.data)
        
        out = Tensor(out_data, requires_grad=requires_grad,
                            is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'sigmoid'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.sigmoid_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.sigmoid_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return Tensor(grad_a, device=grad_output.device), None


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.tanh_forward(a.data)
        else:
            out_data = ops_gpu.tanh_forward(ctx.cl_ctx, ctx.cl_queue, a.data)
        
        out = Tensor(out_data, requires_grad=requires_grad,
                            is_leaf=is_leaf, device=a.device)
        out.children = [a]
        out.op = 'tanh'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        if grad_output.device == Device.CPU:
            grad_a = ops_cpu.tanh_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.tanh_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return Tensor(grad_a, device=grad_output.device), None


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, stride):
        requires_grad = x.requires_grad
        is_leaf = not x.requires_grad

        if x.device == Device.CPU:
            out, x_reshaped = ops_cpu.conv1d_forward(x.data, weight.data, stride)
            ctx.x_reshaped = x_reshaped
        else:
            out = ops_gpu.conv1d_forward(ctx.cl_ctx, ctx.cl_queue, x.data, weight.data, stride)
        
        ctx.save_for_backward(x, weight)
        ctx.stride = stride

        out = Tensor(out, device=x.device, requires_grad=requires_grad, is_leaf=is_leaf)
        out.children = [x, weight]
        out.op = 'conv1d'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride = ctx.stride

        if grad_output.device == Device.CPU:
            x_reshaped = ctx.x_reshaped
            grad_x, grad_weight = ops_cpu.conv1d_backward(grad_output.data, x, x_reshaped, weight.data, stride)
        else:
            grad_x, grad_weight = ops_gpu.conv1d_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                                          x.data, weight.data, stride)

        return Tensor(grad_x, device=grad_output.device), Tensor(grad_weight, device=grad_output.device)
        

class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, stride):
        requires_grad = x.requires_grad
        is_leaf = not requires_grad

        if x.device == Device.CPU:
            out, x_reshaped = ops_cpu.conv2d_forward(x.data, weight, stride)
            ctx.x_reshaped = x_reshaped
        else:
            out = ops_gpu.conv2d_forward(ctx.cl_ctx, ctx.cl_queue, x.data, weight.data, stride)

        ctx.save_for_backward(x, weight)
        ctx.stride = stride

        out = Tensor(out, requires_grad=requires_grad, is_leaf=is_leaf, device=x.device)
        out.children = [x, weight]
        out.op = 'conv2d'
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride = ctx.stride

        if grad_output.device == Device.CPU:
            x_reshaped = ctx.x_reshaped
            grad_x, grad_weight = ops_cpu.conv2d_backward(grad_output.data, x, x_reshaped, weight.data, stride)
        else:
            grad_x, grad_weight = ops_gpu.conv2d_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                                                     x.data, weight.data, stride)

        grad_x = Tensor(grad_x, device=grad_output.device)
        grad_weight = Tensor(grad_weight, device=grad_output.device)

        return grad_x, grad_weight