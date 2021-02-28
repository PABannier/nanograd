import numpy as np
from nanograd.nn.conv_ops import get_conv1d_output_size, get_conv2d_output_size
from nanograd.autograd import Function

# *************************************
# ************** Helpers **************
# *************************************

def sigmoid(x:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def inner_slice(a, indices):
    """
        Helper function to slice a Tensor

        Args:
            a (np.ndarray): array to slice
            indices (list): list of indices 
        
        ..note: Length must match the number of dimensions of x
    """
    padding = [(max(0, -p[0]), max(0, p[1]-a.shape[i])) for i, p in enumerate(indices)]
    a = np.pad(a, padding, mode="constant")
    slices = [(p[0]+padding[i][0], p[1]+padding[i][0]) for i, p in enumerate(indices)]
    return a[tuple([slice(x[0], x[1], None) for x in slices])]

# *************************************
# *********** Forward passes **********
# *************************************

class OneHot(Function):
    @staticmethod
    def forward(ctx, input, num_classes):
        idx = input.astype(int)
        out = np.zeros((idx.shape[0], num_classes))
        out[np.arange(len(out)), idx] = 1
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, input, axis):
        ctx.save_for_backward(axis)
        return np.expand_dims(input, axis)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.saved_tensors[0]
        return grad_output.squeeze(axis)


class Squeeze(Function):
    @staticmethod
    def forward(ctx, input, axis):
        ctx.save_for_backward(axis)
        return np.squeeze(input, axis)
    
    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.saved_tensors
        return np.expand_dims(grad_output, axis)


class Slice(Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(input.shape, indices)
        return inner_slice(input, indices)
    
    @staticmethod
    def backward(ctx, grad_output):
        shape, indices = ctx.saved_tensors
        indices = [(0 - p[0], grad_output.shape[i] + (shape[i] - p[1])) for i, p in enumerate(indices)]
        return inner_slice(grad_output, indices)


class Transpose(Function):
    @staticmethod
    def forward(ctx, input):
        return input.T
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.T


class Reshape(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.save_for_backward(input.shape)
        return input.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.saved_tensors[0]
        return grad_output.reshape(shape)


class Max(Function):
    @staticmethod
    def forward(ctx, input, axis):
        axis = [axis] if isinstance(axis, int) else axis
        out = np.amax(input, axis=None if axis is None else tuple(axis), keepdims=True)
        ctx.save_for_backward(input, axis, out)
        if axis is not None:
            out = out.reshape([input.shape[i] for i in range(len(input.shape)) if i not in axis])
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, axis, out = ctx.saved_tensors
        shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
        ret2 = (input == out.reshape(shape))
        div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
        return ret2 * (grad_output.reshape(shape)).data / div

    
class Min(Function):
    @staticmethod
    def forward(ctx, input, axis):
        axis = [axis] if isinstance(axis, int) else axis
        out = np.amin(input, axis=None if axis is None else tuple(axis), keepdims=True)
        ctx.save_for_backward(input, axis, out)
        if axis is not None:
            out = out.reshape([input.shape[i] for i in range(len(input.shape)) if i not in axis])
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, axis, out = ctx.saved_tensors
        shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
        ret2 = (input == out.reshape(shape))
        div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
        return ret2 * (grad_output.reshape(shape)).data / div


class Sum(Function):
    @staticmethod
    def forward(ctx, input, axis=None):
        ctx.save_for_backward(input, axis)
        if axis is None:
            return np.array([input.sum()])
        return input.sum(axis=axis)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, axis = ctx.saved_tensors
        axis = [axis] if type(axis) == int else axis
        shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
        return grad_output.reshape(shape) + np.zeros_like(input) # Useful for broadcasting


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        grad_a = grad_output * np.ones(a_shape)
        grad_b = grad_output * np.ones(b_shape)
        return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b


class Log(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.log(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return grad_output / input


class Exp(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.exp(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return grad_output * np.exp(input)


class Neg(Function):
    @staticmethod
    def forward(ctx, input):
        return -input
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Pow(Function):
    @staticmethod
    def forward(ctx, input, power):
        ctx.save_for_backward(input, power)
        return input ** power

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.saved_tensors
        return unbroadcast(power * (input ** (power-1.0)) * grad_output, input.shape), \
               unbroadcast((input ** power) * np.log(input) * grad_output, power.shape)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return (input >= 0) * grad_output


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sigmoid(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return grad_output * sigmoid(input) * (1 - sigmoid(input))


class Tanh(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.tanh(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return grad_output * (1 - np.power(np.tanh(input), 2))


class Conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride):
        if not isinstance(stride, int):
            stride = int(stride[0])
        batch_size, in_channel, signal_length = input.shape
        num_filters, _, kernel_length = weight.shape
        output_length = get_conv1d_output_size(signal_length, kernel_length, stride, 0)

        ctx.save_for_backward(input, weight, stride)

        stride_shape = (signal_length, 1, in_channel * signal_length, stride)
        strides = input.data.itemsize * np.array(stride_shape)

        cols = np.lib.stride_tricks.as_strided(
            x=input,
            strides=strides,
            shape=(in_channel, kernel_length, batch_size, output_length),
            writeable=False
        )

        cols = cols.transpose(2, 0, 1, 3)
        weight = weight.transpose(1, 2, 0) 

        ret = np.tensordot(cols, weight, axes=[(1, 2), (0, 1)])
        ret = ret.transpose(0, 2, 1)

        ctx.save_for_backward(cols.transpose(0, 1, 3, 2))
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, stride, input_reshaped = ctx.saved_tensors

        batch_size, in_channel, signal_length = input.shape
        num_filters, _, kernel_length = weight.shape
        _, _, output_length = grad_output.shape

        #grad_weight = np.einsum('ikX, ijXx -> kjx', grad_output, x_reshaped) SLOWER than using tensordot
        grad_weight = np.tensordot(grad_output, input_reshaped, axes=[(0, 2), (0, 2)])

        grad_x = np.zeros((batch_size, in_channel, signal_length), dtype=grad_output.dtype)

        for k in range(output_length):
            X = k % output_length
            iX = X * stride

            #grad_x[:, :, iX:iX+kernel_length] += np.einsum('ik, kjy->ijy', grad_output[:, :, X], weight) #SLOWER than using tensordot
            grad_x[:, :, iX:iX+kernel_length] += np.tensordot(grad_output[:, :, X], weight, axes=[(1), (0)])
        
        grad_x = grad_x.reshape((batch_size, in_channel, signal_length))

        return grad_x, grad_weight


class Conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride):
        if not isinstance(stride, int):
            stride = int(stride[0])
        batch_size, in_channel, im_height, im_width = input.shape
        num_filters, _, kernel_height, kernel_width = weight.shape
        output_height, output_width = get_conv2d_output_size(im_height, im_width, (kernel_height, kernel_width), stride, 0)

        ctx.save_for_backward(weight, stride)

        strides = (im_height * im_width, im_width, 1, in_channel * im_height * im_height, stride * im_width, stride)
        strides = input.itemsize * np.array(strides)

        cols = np.lib.stride_tricks.as_strided(
            x=input,
            shape=(in_channel, kernel_height, kernel_width, batch_size, output_height, output_width),
            strides=strides,
            writeable=False
        )

        cols = cols.transpose(3, 0, 1, 2, 4, 5)
        weight = weight.transpose(1, 2, 3, 0)

        #jiyxYX,iyxk -> jYXk -> jkYX
        ret = np.tensordot(cols, weight, axes=[(1, 2, 3), (0, 1, 2)])
        ret = ret.transpose(0, 3, 1, 2)

        ctx.save_for_backward(input, cols.transpose(0, 1, 4, 5, 2, 3))

        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        weight, stride, input, input_reshaped = ctx.saved_tensors

        batch_size, in_channel, im_height, im_width = input.shape
        num_filters, _, kernel_height, kernel_width = weight.shape
        _, _, output_height, output_width = grad_output.shape
        
        #grad_weight = np.einsum('ikYX, ijYXyx -> kjyx', grad_output, x_reshaped) SLOWER than using tensordot 
        grad_weight = np.tensordot(grad_output, input_reshaped, axes=[(0,2,3),(0,2,3)])

        grad_x = np.zeros((batch_size, in_channel, im_height, im_width), dtype=grad_output.dtype)

        for k in range(output_height * output_width):
            X, Y = k % output_width, k // output_width
            iX, iY = X * stride, Y * stride

            # grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.einsum('ik,kjyx->ijyx', grad_output[:,:,Y,X], weight) 
            # SLOWER than using tensordot
            grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.tensordot(grad_output[:,:,Y,X], weight, axes=[(1), (0)])

        grad_x = grad_x.reshape((batch_size, in_channel, im_height, im_width))

        return grad_x, grad_weight