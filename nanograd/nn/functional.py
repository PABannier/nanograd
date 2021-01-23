import numpy as np

from nanograd.tensor import Tensor
from nanograd.device import Device
import nanograd.nn.ops_cpu as ops_cpu
import nanograd.nn.ops_gpu as ops_gpu
from nanograd.autograd_engine import Function
from nanograd.nn.conv_ops import (get_conv1d_output_size, get_conv2d_output_size, 
                                  get_im2col_indices, col2im)


def cross_entropy(predicted:Tensor, target:Tensor) -> Tensor:
    r"""
        Calculates Cross Entropy Loss between logits and true labels.
        Used in the CrossEntropy module.

        Args:
            predicted (Tensor): (batch_size, num_classes) logits
            target (Tensor): (batch_size,) true labels

        Returns:
            Tensor: the loss as a float, in a tensor of shape ()
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
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for OneHot must be tensor: {}".format(type(a).__name__))

        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        if a.device == Device.CPU:
            out = ops_cpu.one_hot_encoding_op(a.data,  num_classes)
        else:
            out = ops_gpu.one_hot_encoding_op(ctx.cl_ctx, ctx.cl_queue, a.data, num_classes)
        
        return Tensor(out, device=a.device, requires_grad=requires_grad,
                        is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        # No backward pass since one-hot encoding is applied to a target
        # tensor whose gradient is None
        return None


class Slice(Function):
    @staticmethod
    def forward(ctx, a, indices=None):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Slice must be tensor: {}".format(type(a).__name__))

        ctx.shape, ctx.indices = a.shape, indices

        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        print(a.device)

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
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))

        ctx.shape = a.shape
        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.reshape_forward(a.data, shape)
        else:
            out_data = ops_gpu.reshape_forward(ctx.cl_ctx, ctx.cl_queue, a, shape)
        
        out = Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
            
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
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Max must be tensor: {}".format(type(a).__name__))

        axis = [axis] if type(axis) == int else axis

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == Device.CPU:
            out_data = ops_cpu.max_forward(a.data, axis, keepdims)
        else:
            out_data = ops_gpu.max_forward(ctx.cl_ctx, ctx.cl_queue, a.data, axis, keepdims)
        
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
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Min must be tensor: {}".format(type(a).__name__))

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
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))

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
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
            
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
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

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
        if not type(a).__name__ == "Tensor":
            raise Exception(f"Only sum of tensor is supported. Got: {type(a).__name__}")

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
        if not type(a).__name__ == "Tensor":
            raise Exception(f"Only neg of tensor is supported. Got: {type(a).__name__}")

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
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

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
        if not type(a).__name__ == "Tensor":
            raise Exception("Only power of tensor is supported")
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
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Only tensors can be multiplied element-wise")

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
        if not type(a).__name__ == 'Tensor':
            raise Exception("ReLU can only be applied to tensors")

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
        if not type(a).__name__ == "Tensor":
            raise Exception("Sigmoid can only be applied to tensors")

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
        if not type(a).__name__ == "Tensor":
            raise Exception("Tanh can only be applied to tensors")

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
    def forward(ctx, x, weight, bias, stride, pad):
        """
            The forward/backward of a Conv1d Layer in the comp graph.
            
            Args:
                x (Tensor): (batch_size, in_channel, input_size) input data
                weight (Tensor): (out_channel, in_channel, kernel_length)
                bias (Tensor): (out_channel,)
                stride (int): Stride of the convolution
                pad (int): Padding for the convolution
            
            Returns:
                Tensor: (batch_size, out_channel, output_size) output data
        """
        N, C, L = x.shape
        F, _, KL = weight.shape
        OL = get_conv1d_output_size(L, KL, stride, pad)

        L += 2 * pad
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (pad, pad)), mode="constant")

        stride_shape = (L, 1, C * L, stride)
        strides = x.data.itemsize * np.array(stride_shape)
        x_strides = np.lib.stride_tricks.as_strided(
            x=x_padded,
            strides=strides,
            shape=(C, KL, N, OL),
            writeable=False
        )

        x_cols = np.ascontiguousarray(x_strides)
        x_cols.shape = (C * KL, N * OL)

        out = weight.data.reshape(F, -1) @ x_cols + bias.data.reshape(-1, 1)
        out.shape = (F, N, OL)
        out = out.transpose(1, 0, 2)

        ctx.save_for_backward(x, weight, bias)
        ctx.x_cols = x_cols
        ctx.stride, ctx.pad = stride, pad

        out = Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
        out.children = [x, weight, bias]
        out.op = 'conv1d'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        x_cols = ctx.x_cols
        stride, pad = ctx.stride, ctx.pad

        N, C, L = x.shape
        F, _, KL = weight.shape
        _, _,  OL = grad_output.shape

        grad_bias = np.sum(grad_output.data, axis=(0, 2))
        grad_bias = Tensor(grad_bias)

        grad_out_reshaped = grad_output.data.transpose(1, 2, 0).reshape(F, -1)
        grad_weight = (grad_out_reshaped @ x_cols.T).reshape(weight.shape)
        grad_weight = Tensor(grad_weight)

        grad_x_cols = weight.data.reshape(F, -1).T @ grad_out_reshaped
        grad_x_cols.shape = (C, KL, N, OL)
        grad_x = col2im(grad_x_cols, x.shape, 1, KL, pad, stride)
        grad_x = Tensor(grad_x)

        return grad_x, grad_weight, grad_bias
        

class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, pad):
        """
            The forward/backward of a Conv2d Layer in the comp graph.
            
            Args:
                x (Tensor): (batch_size, in_channel, input_height, input_width) input data
                weight (Tensor): (out_channel, in_channel, kernel_height, kernel_width)
                bias (Tensor): (out_channel,)
                stride (int): Stride of the convolution
                pad (int): Padding for the convolution
            
            Returns:
                Tensor: (batch_size, out_channel, output_height, output_width) output data
        """
        N, C, H, W = x.shape
        F, _, HH, WW = weight.shape
        OH, OW = get_conv2d_output_size(H, W, (HH, WW), stride, pad)

        x_padded = np.pad(x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

        H += 2 * pad
        W += 2 * pad
        out = np.zeros((N, F, OH, OW))

        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.data.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(
            x=x_padded,
            shape=(C, HH, WW, N, OH, OW),
            strides=strides,
            writeable=False
        )
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * OH * OW)

        res = weight.data.reshape(F, -1) @ x_cols + bias.data.reshape(-1, 1)
        res.shape = (F, N, OH, OW)
        out = res.transpose(1, 0, 2, 3)

        ctx.save_for_backward(x, weight, bias)
        ctx.x_cols = x_cols
        ctx.stride, ctx.pad = stride, pad

        out = Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
        out.children = [x, weight, bias]
        out.op = 'conv2d'
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        x_cols = ctx.x_cols
        stride, pad = ctx.stride, ctx.pad

        N, C, H, W = x.shape
        F, _, HH, WW = weight.shape
        _, _,  OH, OW = grad_output.shape

        grad_bias = np.sum(grad_output.data, axis=(0, 2, 3))
        grad_bias = Tensor(grad_bias)

        grad_out_reshaped = grad_output.data.transpose(1, 2, 3, 0).reshape(F, -1)
        grad_weight = (grad_out_reshaped @ x_cols.T).reshape(weight.shape)
        grad_weight = Tensor(grad_weight)
        
        grad_x_cols = weight.data.reshape(F, -1).T @ grad_out_reshaped
        grad_x_cols.shape = (C, HH, WW, N, OH, OW)
        grad_x = col2im(grad_x_cols, x.shape, HH, WW, pad, stride) # Needs to be optimized
        grad_x = Tensor(grad_x)

        return grad_x, grad_weight, grad_bias