import numpy as np


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

def max_forward(a, axis):
    out = np.amax(a, axis=None if axis is None else tuple(axis), keepdims=True) 
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