import numpy as np
import tensor
from autograd_engine import Function
from nn.conv_ops import (get_conv1d_output_size, get_conv2d_output_size, 
                         get_im2col_indices, col2im)
import nn.ops_cpu as ops_cpu
import nn.ops_gpu as ops_gpu


def inner_slice(a, indices):
    padding = [(max(0, -p[0]), max(0, p[1]-a.shape[i])) for i, p in enumerate(indices)]
    a = np.pad(a, padding, mode="constant")
    slices = [(p[0]+padding[i][0], p[1]+padding[i][0]) for i, p in enumerate(indices)]
    return a[tuple([slice(x[0], x[1], None) for x in slices])]

def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def cross_entropy(predicted, target):
    r"""
        Calculates Cross Entropy Loss (XELoss) between logits and true labels.
        For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

        Args:
            predicted (Tensor): (batch_size, num_classes) logits
            target (Tensor): (batch_size,) true labels

        Returns:
            Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape
    labels = to_one_hot(target, num_classes)

    a = tensor.Tensor(np.amax(predicted.data, axis=1)).reshape((batch_size, 1))
    log_softmax = predicted - a - tensor.Tensor.sum((predicted - a).exp(), axis=1).log().reshape((batch_size, 1))

    nll_loss = tensor.Tensor.sum(log_softmax * labels) / tensor.Tensor(-batch_size)

    return nll_loss


def to_one_hot(arr, num_classes:int):
    """
        Converts a tensor of classes to one-hot, useful in XELoss

        Example:
        >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
        [[0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0]]

        Args:
            arr (Tensor): Condensed tensor of label indices
            num_classes (int): Number of possible classes in dataset
                            For instance, MNIST would have `num_classes==10`
        Returns:
            Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)


class Slice(Function):
    @staticmethod
    def forward(ctx, a, indices=None):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Slice must be tensor: {}".format(type(a).__name__))

        ctx.shape, ctx.indices = a.shape, indices

        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        if a.device == tensor.Device.CPU:
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
        indices = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i, p in enumerate(fwd_indices)]
        out = inner_slice(grad_output.data, indices)
        return tensor.Tensor(out), None


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        
        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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
        return tensor.Tensor(grad_output.data.T), None


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))

        ctx.shape = a.shape
        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
            out_data = ops_cpu.reshape_forward(a.data, shape)
        else:
            out_data = ops_gpu.reshape_forward(ctx.cl_ctx, ctx.cl_queue, a, shape)
        
        out = tensor.Tensor(out_data, requires_grad=requires_grad, 
                            is_leaf=is_leaf, device=a.device)
            
        out.children = [a]
        out.op = 'reshape'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Max must be tensor: {}".format(type(a).__name__))

        axis = [axis] if type(axis) == int else axis

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
            out_data = ops_cpu.max_forward(a.data, axis, keepdims)
        else:
            out_data = ops_gpu.max_forward(ctx.cl_ctx, ctx.cl_queue, a.data, axis, keepdims)
        
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
        
        shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
        ret2 = (inp.data == out.reshape(shape))
        div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
        out = ret2 * (grad_output.reshape(shape)).data / div
        return tensor.Tensor(out), None


class Min(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Min must be tensor: {}".format(type(a).__name__))

        axis = [axis] if type(axis) == int else axis

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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
        
        shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
        ret2 = (inp.data == out.reshape(shape))
        div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
        out = ret2 * (grad_output.reshape(shape)).data / div
        return tensor.Tensor(out), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))

        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a = ops_cpu.log_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.log_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return tensor.Tensor(grad_a), None
    

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
            
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a = ops_cpu.exp_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.exp_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return tensor.Tensor(grad_a), None


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a, grad_b = ops_cpu.add_backward(grad_output.data, a.shape, b.shape)
        else:
            grad_a, grad_b = ops_gpu.add_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data,
                                                  a.shape, b.shape)

        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == "Tensor":
            raise Exception(f"Only sum of tensor is supported. Got: {type(a).__name__}")

        ctx.axis = axis
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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

        axis = [axis] if type(axis) == int else axis
        shape = [1 if axis is None or i in axis else a.shape[i] for i in range(len(a.shape))]
        
        grad_a = grad_output.data.reshape(shape) + np.zeros_like(a.data)
        
        return tensor.Tensor(grad_a), None, None


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception(f"Only neg of tensor is supported. Got: {type(a).__name__}")

        requires_grad = a.requires_grad

        if a.device == tensor.Device.CPU:
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
        if grad_output.device == tensor.Device.CPU:
            grad = ops_cpu.neg_backward(grad_output.data)
        else:
            grad = ops_gpu.neg_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data)
        return tensor.Tensor(grad), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        if a.shape[1] != b.shape[0]:
            raise Exception(f"Shapes don't match: {a.shape}, {b.shape}")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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

        grad_a = np.matmul(grad_output.data, np.transpose(b.data))
        grad_b = np.matmul(np.transpose(a.data), grad_output.data)

        grad_a = tensor.Tensor(grad_a)
        grad_b = tensor.Tensor(grad_b)

        return grad_a, grad_b


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

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a = ops_cpu.pow_backward(grad_output.data, a.data, exp)
        else:
            grad_a = ops_gpu.pow_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data, exp)

        return tensor.Tensor(grad_a), None


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Only tensors can be multiplied element-wise")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        if a.device == tensor.Device.CPU:
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

        grad_a = grad_output.data * b.data
        grad_b = grad_output.data * a.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("ReLU can only be applied to tensors")

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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

        if grad_output.device == tensor.Device.CPU:
            grad_a = ops_cpu.relu_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.relu_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return tensor.Tensor(grad_a), None


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Sigmoid can only be applied to tensors")

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a = ops_cpu.sigmoid_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.sigmoid_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return tensor.Tensor(grad_a), None


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Tanh can only be applied to tensors")

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        is_leaf = not requires_grad

        if a.device == tensor.Device.CPU:
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

        if a.device == tensor.Device.CPU:
            grad_a = ops_cpu.tanh_backward(grad_output.data, a.data)
        else:
            grad_a = ops_gpu.tanh_backward(ctx.cl_ctx, ctx.cl_queue, grad_output.data, a.data)

        return tensor.Tensor(grad_a), None


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

        out = tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
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
        grad_bias = tensor.Tensor(grad_bias)

        grad_out_reshaped = grad_output.data.transpose(1, 2, 0).reshape(F, -1)
        grad_weight = (grad_out_reshaped @ x_cols.T).reshape(weight.shape)
        grad_weight = tensor.Tensor(grad_weight)

        grad_x_cols = weight.data.reshape(F, -1).T @ grad_out_reshaped
        grad_x_cols.shape = (C, KL, N, OL)
        grad_x = col2im(grad_x_cols, x.shape, 1, KL, pad, stride)
        grad_x = tensor.Tensor(grad_x)

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

        out = tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
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
        grad_bias = tensor.Tensor(grad_bias)

        grad_out_reshaped = grad_output.data.transpose(1, 2, 3, 0).reshape(F, -1)
        grad_weight = (grad_out_reshaped @ x_cols.T).reshape(weight.shape)
        grad_weight = tensor.Tensor(grad_weight)
        
        grad_x_cols = weight.data.reshape(F, -1).T @ grad_out_reshaped
        grad_x_cols.shape = (C, HH, WW, N, OH, OW)
        grad_x = col2im(grad_x_cols, x.shape, HH, WW, pad, stride) # Needs to be optimized
        grad_x = tensor.Tensor(grad_x)

        return grad_x, grad_weight, grad_bias