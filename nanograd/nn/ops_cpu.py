import numpy as np
from nanograd.nn.conv_ops import get_conv1d_output_size, get_conv2d_output_size

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

def one_hot_encoding(a, num_classes):
    idx = a.astype(int)
    out = np.zeros((idx.shape[0], num_classes))
    out[np.arange(len(out)), idx] = 1
    return out

# *************************************
# *********** Forward passes **********
# *************************************

def unsqueeze_forward(a, axis):
    return np.expand_dims(a, axis)

def squeeze_forward(a, axis):
    return np.squeeze(a, axis)

def slice_forward(a, indices):
    return inner_slice(a, indices)

def transpose_forward(a):
    return a.T

def reshape_forward(a, shape):
    return a.reshape(shape)

def max_forward(a, axis):
    out = np.amax(a, axis=None if axis is None else tuple(axis), keepdims=True) 
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out

def min_forward(a, axis):
    out = np.amin(a, axis=None if axis is None else tuple(axis), keepdims=True)
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out

def sum_forward(a, axis):
    if axis is None:
        return np.array(a.sum())
    return a.sum(axis=axis)

def add_forward(a, b):
    return a + b

def mul_forward(a, b):
    return a * b

def matmul_forward(a, b):
    return a @ b

def log_forward(a):
    return np.log(a)

def exp_forward(a):
    return np.exp(a)

def neg_forward(a):
    return -a

def pow_forward(a, exp):
    return a ** exp

def relu_forward(a):
    return np.maximum(a, 0)

def sigmoid_forward(a):
    return 1.0 / (1.0 + np.exp(-a))

def tanh_forward(a):
    return np.tanh(a)

def conv1d_forward(a, weight, stride):
    batch_size, in_channel, signal_length = a.shape
    num_filters, _, kernel_length = weight.shape
    output_length = get_conv1d_output_size(signal_length, kernel_length, stride, 0)

    stride_shape = (signal_length, 1, in_channel * signal_length, stride)
    strides = a.data.itemsize * np.array(stride_shape)

    cols = np.lib.stride_tricks.as_strided(
        x=a,
        strides=strides,
        shape=(in_channel, kernel_length, batch_size, output_length),
        writeable=False
    )

    cols = cols.transpose(2, 0, 1, 3)
    weight = weight.transpose(1, 2, 0) 

    ret = np.tensordot(cols, weight, axes=[(1, 2), (0, 1)])
    ret = ret.transpose(0, 2, 1)
    return ret, cols.transpose(0, 1, 3, 2) 


def conv2d_forward(a, weight, stride):
    batch_size, in_channel, im_height, im_width = a.shape
    num_filters, _, kernel_height, kernel_width = weight.shape
    output_height, output_width = get_conv2d_output_size(im_height, im_width, (kernel_height, kernel_width), stride, 0)

    strides = (im_height * im_width, im_width, 1, in_channel * im_height * im_height, stride * im_width, stride)
    strides = a.itemsize * np.array(strides)

    cols = np.lib.stride_tricks.as_strided(
        x=a,
        shape=(in_channel, kernel_height, kernel_width, batch_size, output_height, output_width),
        strides=strides,
        writeable=False
    )

    cols = cols.transpose(3, 0, 1, 2, 4, 5)
    weight = weight.transpose(1, 2, 3, 0)

    #jiyxYX,iyxk -> jYXk -> jkYX
    ret = np.tensordot(cols, weight, axes=[(1, 2, 3), (0, 1, 2)])
    ret = ret.transpose(0, 3, 1, 2)
    return ret, cols.transpose(0, 1, 4, 5, 2, 3) 

# *************************************
# ********** Backward passes **********
# *************************************

def unsqueeze_backward(grad_output, axis):
    return grad_output.squeeze(axis)

def squeeze_backward(grad_output, axis):
    return np.expand_dims(grad_output, axis)

def add_backward(grad_output, a_shape, b_shape):
    grad_a = np.ones(a_shape) * grad_output.data
    grad_b = np.ones(b_shape) * grad_output.data
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)

def mul_backward(grad_output, a, b):
    grad_a = grad_output * b
    grad_b = grad_output * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)

def matmul_backward(grad_output, a, b):
    grad_a = np.matmul(grad_output, b.T)
    grad_b = np.matmul(a.T, grad_output)
    return grad_a, grad_b

def log_backward(grad_output, a):
    return grad_output / a

def exp_backward(grad_output, a):
    return grad_output * np.exp(a)

def neg_backward(grad_output):
    return -grad_output

def pow_backward(grad_output, a, exp):
    return exp * (a ** (exp-1)) * grad_output

def relu_backward(grad_output, a):
    return grad_output * (a >= 0)

def sigmoid_backward(grad_output, a):
    return grad_output * sigmoid(a) * (1 - sigmoid(a))

def tanh_backward(grad_output, a):
    return grad_output * (1 - np.power(np.tanh(a), 2))

def slice_backward(grad_output, shape, fwd_indices):
    indices = [(0 - p[0], grad_output.shape[i] + (shape[i] - p[1])) for i, p in enumerate(fwd_indices)]
    return inner_slice(grad_output, indices)

def transpose_backward(grad_output):
    return grad_output.T

def reshape_backward(grad_output, shape):
    return grad_output.reshape(shape)

def max_backward(grad_output, inp, out, axis):
    shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
    ret2 = (inp == out.reshape(shape))
    div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
    return ret2 * (grad_output.reshape(shape)).data / div

def min_backward(grad_output, inp, out, axis):
    shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
    ret2 = (inp == out.reshape(shape))
    div = ret2.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
    return ret2 * (grad_output.reshape(shape)).data / div

def sum_backward(grad_output, a, axis):
    axis = [axis] if type(axis) == int else axis
    shape = [1 if axis is None or i in axis else a.shape[i] for i in range(len(a.shape))]
    return grad_output.reshape(shape) + np.zeros_like(a) # Useful for broadcasting

def conv1d_backward(grad_output, x, x_reshaped, weight, stride):
    batch_size, in_channel, signal_length = x.shape
    num_filters, _, kernel_length = weight.shape
    _, _, output_length = grad_output.shape

    #grad_weight = np.einsum('ikX, ijXx -> kjx', grad_output, x_reshaped) SLOWER than using tensordot
    grad_weight = np.tensordot(grad_output, x_reshaped, axes=[(0, 2), (0, 2)])

    grad_x = np.zeros((batch_size, in_channel, signal_length), dtype=grad_output.dtype)

    for k in range(output_length):
        X = k % output_length
        iX = X * stride

        #grad_x[:, :, iX:iX+kernel_length] += np.einsum('ik, kjy->ijy', grad_output[:, :, X], weight) SLOWER than using tensordot
        grad_x[:, :, iX:iX+kernel_length] += np.tensordot(grad_output[:, :, X], weight, axes=[(1), (0)])
    
    grad_x = grad_x.reshape((batch_size, in_channel, signal_length))

    return grad_x, grad_weight
        
def conv2d_backward(grad_output, x, x_reshaped, weight, stride):
    batch_size, in_channel, im_height, im_width = x.shape
    num_filters, _, kernel_height, kernel_width = weight.shape
    _, _, output_height, output_width = grad_output.shape
    
    #grad_weight = np.einsum('ikYX, ijYXyx -> kjyx', grad_output, x_reshaped) SLOWER than using tensordot 
    grad_weight = np.tensordot(grad_output, x_reshaped, axes=[(0,2,3),(0,2,3)])

    grad_x = np.zeros((batch_size, in_channel, im_height, im_width), dtype=grad_output.dtype)

    for k in range(output_height * output_width):
        X, Y = k % output_width, k // output_width
        iX, iY = X * stride, Y * stride

        # grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.einsum('ik,kjyx->ijyx', grad_output[:,:,Y,X], weight) 
        # SLOWER than using tensordot
        grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.tensordot(grad_output[:,:,Y,X], weight, axes=[(1), (0)])

    grad_x = grad_x.reshape((batch_size, in_channel, im_height, im_width))

    return grad_x, grad_weight