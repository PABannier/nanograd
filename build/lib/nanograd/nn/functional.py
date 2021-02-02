import numpy as np

#from nanograd.tensor import tensor.Tensor
import nanograd.tensor as tensor
from nanograd.device import Device
import nanograd.nn.ops_cpu as ops_cpu
import nanograd.nn.ops_gpu as ops_gpu
from nanograd.autograd_engine import Function
from nanograd.nn.conv_ops import (get_conv1d_output_size, get_conv2d_output_size, 
                                  get_im2col_indices, col2im)


def cross_entropy(predicted:tensor.Tensor, target:tensor.Tensor) -> tensor.Tensor:
    """Calculates Cross Entropy Loss between logits and true labels.
       Used in the CrossEntropy module

    Args:
        predicted (tensor.Tensor): Logits
        target (tensor.Tensor): Target classes

    Returns:
        tensor.Tensor: Loss in a tensor.Tensor of shape ()
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
        
        return tensor.Tensor(out, device=a.device, requires_grad=requires_grad,
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

        return tensor.Tensor(out, device=a.device, requires_grad=requires_grad,
                      is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis

        if grad_output.device == Device.CPU:
            grad = ops_cpu.unsqueeze_backward(grad_output.data, axis)
        else:
            grad = ops_gpu.unsqueeze_backward(ctx.cl_ctx, ctx.cl_queue, grad_output, axis)
        
        return tensor.Tensor(grad, device=grad_output.device), None


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
        
        return tensor.Tensor(out, device=a.device, requires_grad=requires_grad,
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
        
        return tensor.Tensor(grad, device=grad_output.device), None


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
        
        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        
        return tensor.Tensor(grad, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad, device=grad_output.device), None


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
        
        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
            
        return tensor.Tensor(grad, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad, device=grad_output.device), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.log_forward(a.data)
        else:
            out_data = ops_gpu.log_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad_a, device=grad_output.device), None
    

class Exp(Function):
    @staticmethod
    def forward(ctx, a):   
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.exp_forward(a.data)
        else:
            out_data = ops_gpu.exp_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad_a, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        return tensor.Tensor(grad_a, device=grad_output.device), tensor.Tensor(grad_b, device=grad_output.device)


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        
        return tensor.Tensor(grad, device=grad_output.device), None, None


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        requires_grad = a.requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.neg_forward(a.data)
        else:
            out_data = ops_gpu.neg_forward(ctx.cl_ctx, ctx.cl_queue, a.data)

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        return tensor.Tensor(grad, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        return tensor.Tensor(grad_a, device=grad_output.device), tensor.Tensor(grad_b, device=grad_output.device)


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad_a, device=grad_output.device), None


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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
        
        return tensor.Tensor(grad_a, device=grad_output.device), tensor.Tensor(grad_b, device=grad_output.device)


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

        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
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

        return tensor.Tensor(grad_a, device=grad_output.device), None


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
        
        out = tensor.Tensor(out_data, requires_grad=requires_grad,
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

        return tensor.Tensor(grad_a, device=grad_output.device), None


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
        
        out = tensor.Tensor(out_data, requires_grad=requires_grad,
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

        return tensor.Tensor(grad_a, device=grad_output.device), None


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, stride):
        """
            The forward/backward of a Conv1d Layer in the comp graph.
            
            Args:
                x (tensor.Tensor): (batch_size, in_channel, input_size) input data
                weight (tensor.Tensor): (out_channel, in_channel, kernel_length)
                stride (int): Stride of the convolution
            
            Returns:
                tensor.Tensor: (batch_size, out_channel, output_size) output data
        """

        requires_grad = x.requires_grad
        is_leaf = not x.requires_grad

        if x.device == Device.CPU:
            out, x_cols = ops_cpu.conv1d_forward(x.data, weight.data, stride)
            ctx.x_cols = x_cols
        else:
            out = ops_gpu.conv1d_forward(ctx.cl_ctx, ctx.cl_queue, x.data, weight.data, stride)
        
        ctx.save_for_backward(x, weight)
        ctx.stride = stride

        out = tensor.Tensor(out, device=x.device, requires_grad=requires_grad, is_leaf=is_leaf)
        out.children = [x, weight]
        out.op = 'conv1d'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride = ctx.stride

        if grad_output.device == Device.CPU:
            x_cols = ctx.x_cols
            grad_x, grad_weight = ops_cpu.conv1d_backward(grad_output.data, x.data, x_cols, weight.data, stride)
        else:
            grad_x, grad_weight = ops_gpu.conv1d_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                                          x.data, weight.data, stride)

        return tensor.Tensor(grad_x, device=grad_output.device), tensor.Tensor(grad_weight, device=grad_output.device)
        

class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, stride):
        """
            The forward/backward of a Conv2d Layer in the comp graph.
            
            Args:
                x (tensor.Tensor): (batch_size, in_channel, input_height, input_width) input data
                weight (tensor.Tensor): (out_channel, in_channel, kernel_height, kernel_width)
                stride (int): Stride of the convolution
            
            Returns:
                tensor.Tensor: (batch_size, out_channel, output_height, output_width) output data
        """

        requires_grad = x.requires_grad
        is_leaf = not requires_grad

        if x.device == Device.CPU:
            out, x_cols = ops_cpu.conv2d_forward(x.data, weight, stride)
            ctx.x_cols = x_cols
        else:
            out = ops_gpu.conv2d_forward(ctx.cl_ctx, ctx.cl_queue, x.data, weight.data, stride)

        ctx.save_for_backward(x, weight)
        ctx.stride = stride

        out = tensor.Tensor(out, requires_grad=requires_grad, is_leaf=is_leaf, device=x.device)
        out.children = [x, weight]
        out.op = 'conv2d'
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride = ctx.stride

        if grad_output.device == Device.CPU:
            x_cols = ctx.x_cols
            grad_x, grad_weight = ops_cpu.conv2d_backward(grad_output.data, x, weight, x_cols, stride)
        else:
            grad_x, grad_weight = ops_gpu.conv2d_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, 
                                                                     x.data, weight.data, stride)

        grad_x = tensor.Tensor(grad_x, device=grad_output.device)
        grad_weight = tensor.Tensor(grad_weight, device=grad_output.device)

        return grad_x, grad_weight