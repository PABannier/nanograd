import numpy as np


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

# *************************************
# *********** Forward passes **********
# *************************************

def slice_forward(a, indices):
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

def transpose_forward(a):
    return a.T

def reshape_forward(a, shape):
    return a.reshape(shape)

def max_forward(a, axis, keepdims):
    out = np.amax(a, axis=None if axis is None else tuple(axis), keepdims=True) 
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out

def min_forward(a, axis, keepdims):
    out = np.amin(a, axis=None if axis is None else tuple(axis), keepdims=True)
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out

def sum_forward(a, axis, keepdims):
    if axis is None:
        return np.array(a.sum(keepdims=keepdims))
    return a.sum(axis=axis, keepdims=keepdims)

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

# *************************************
# ********** Backward passes **********
# *************************************

def add_backward(grad_output, a_shape, b_shape):
    grad_a = np.ones(a_shape) * grad_output.data
    grad_b = np.ones(b_shape) * grad_output.data
    return unbroadcast(grad_a, a_shape), unbroadcast(grad_b, b_shape)

def mul_backward(grad_output, a, b):
    grad_a = grad_output * b
    grad_b = grad_output * a
    return unbroadcast(grad_a, a.shape), unbroadcast(grad_b, b.shape)

def matmul_backward(grad_output, a):
    raise NotImplementedError

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

def slice_backward(grad_output, a):
    raise NotImplementedError

def transpose_backward(grad_output, a):
    raise NotImplementedError

def reshape_backward(grad_output, a):
    raise NotImplementedError

def max_backward(grad_output, a):
    raise NotImplementedError

def min_backward(grad_output, a):
    raise NotImplementedError

def sum_backward(grad_output, a):
    raise NotImplementedError

def conv1d_backward(a, weight, bias, stride, pad):
    raise NotImplementedError

def conv2d_backward(a, weight, bias, stride, pad):
    raise NotImplementedError