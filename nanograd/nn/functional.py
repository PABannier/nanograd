import numpy as np
import tensor
from autograd_engine import Function
from nn.conv_ops import (get_conv1d_output_size, get_conv2d_output_size, 
                         get_im2col_indices, im2col, col2im)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
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


def to_one_hot(arr, num_classes):
    """Converts a tensor of classes to one-hot, useful in XELoss

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


def inner_slice(x, indices):
  padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i])) for i, p in enumerate(indices)]
  x = np.pad(x, padding, mode="constant")
  slices = [(p[0]+padding[i][0], p[1]+padding[i][0]) for i, p in enumerate(indices)]
  return x[tuple([slice(x[0], x[1], None) for x in slices])]


class Slice(Function):
    @staticmethod
    def forward(ctx, a, indices):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Slice must be tensor: {}".format(type(a).__name__))

        requires_grad = a.requires_grad
        is_leaf = not a.requires_grad

        out = inner_slice(a.data, indices)
        ctx.shape = a.shape
        ctx.indices = indices

        return tensor.Tensor(out, requires_grad=requires_grad, is_leaf=is_leaf)
    
    @staticmethod
    def backward(ctx, grad_output):
        shape, fwd_indices = ctx.shape, ctx.indices
        indices = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i, p in enumerate(fwd_indices)]
        out = inner_slice(grad_output.data, indices)
        return tensor.Tensor(out),


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad, 
                                    is_leaf=not requires_grad)
        return b

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
        b = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad, 
                                                 is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Max must be tensor: {}".format(type(a).__name__))

        axis = [axis] if type(axis) == int else axis
        out = np.amax(a.data, axis=None if axis is None else tuple(axis), keepdims=True) 

        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.out = out

        if axis is not None:
            out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])

        return tensor.Tensor(out, requires_grad=a.requires_grad, is_leaf=not a.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis, out = ctx.axis, ctx.out
        inp = ctx.saved_tensors[0]
        
        shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
        ret2 = (inp.data == out.reshape(shape))
        div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
        out = ret2 * (grad_output.reshape(shape)).data / div
        return tensor.Tensor(out), 


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))

        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        b = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data), None
    

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
            
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        b = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad, 
                                          is_leaf=not requires_grad)
        return b
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = grad_output.data * np.exp(a.data)
        return tensor.Tensor(grad_a), None
        

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad, 
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == "Tensor":
            raise Exception(f"Only sum of tensor is supported. Got: {type(a).__name__}")

        ctx.axis = axis
        ctx.save_for_backward(a)

        if axis is None:
            out = np.array(a.data.sum(keepdims=keepdims))
        else:
            out = a.data.sum(axis=axis, keepdims=keepdims)

        return tensor.Tensor(out, requires_grad=a.requires_grad, is_leaf=not a.requires_grad)
    
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

        b = tensor.Tensor(-a.data, requires_grad=a.requires_grad,
                                    is_leaf=not a.requires_grad)
        return b
    
    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(-grad_output.data), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        if a.shape[1] != b.shape[0]:
            raise Exception(f"Shapes don't match: {a.shape}, {b.shape}")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(np.matmul(a.data, b.data), requires_grad=requires_grad, \
                                                      is_leaf=not requires_grad)
        return c
    
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

        out = a.data ** exp

        ctx.save_for_backward(a)
        ctx.exp = exp
        c = tensor.Tensor(out, requires_grad=a.requires_grad, \
                               is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        exp = ctx.exp
        a = ctx.saved_tensors[0]
        grad_a = exp * (a.data ** (exp-1)) * grad_output.data

        return tensor.Tensor(grad_a), None


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Only tensors can be multiplied element-wise")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(np.multiply(a.data, b.data), requires_grad=requires_grad, \
                                                        is_leaf=not requires_grad)
        return c
    
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

        c = tensor.Tensor(np.maximum(a.data, 0), requires_grad=a.requires_grad, \
                                                 is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = grad_output.data * (a.data >= 0)

        return tensor.Tensor(grad_a), None


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Sigmoid can only be applied to tensors")

        ctx.save_for_backward(a)

        c = tensor.Tensor(sigmoid(a.data), requires_grad=a.requires_grad, \
                                           is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = grad_output.data * sigmoid(a.data) * (1 - sigmoid(a.data))
        return tensor.Tensor(grad_a), None


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Tanh can only be applied to tensors")

        ctx.save_for_backward(a)

        c = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad, \
                                           is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = grad_output.data * (1 - np.power(np.tanh(a.data), 2))
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

        return tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
    
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
        
        return tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
    
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
        grad_x = col2im(grad_x_cols, x.shape, HH, WW, pad, stride)
        grad_x = tensor.Tensor(grad_x)

        return grad_x, grad_weight, grad_bias