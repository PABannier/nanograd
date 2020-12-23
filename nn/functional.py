import numpy as np
import tensor
from nanograd.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    r"""
        ????????
    """
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

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
        b = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)
        return b
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = grad_output.data * np.exp(a.data)
        return tensor.Tensor(grad_a), None

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Sqrt must be tensor: {}".format(type(a).__name__))

        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        b = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)
        return b
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = np.divide(grad_output.data, 2 * np.sqrt(a.data))
        return tensor.Tensor(grad_a), None

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not(type(a).__name__ == "Tensor" and type(b).__name__  == "Tensor"):
            raise Exception(f"Both args must be  Tensor: {type(a)}, {type(b)}")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = grad_output.data
        grad_b = -grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception(f"Only sum of tensor is supported. Got: {type(a).__name__}")

        ctx.axis = axis
        ctx.shape = a.shape

        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out
        assert grad.shape == ctx.shape

        return tensor.Tensor(grad), None, None

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
        c = tensor.Tensor(np.power(a.data, exp), requires_grad=a.requires_grad, \
                                                  is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        exp = ctx.exp
        a = ctx.saved_tensors[0]

        grad_a = exp * (a.data ** (exp-1)) * grad_output.data

        return grad_a, None

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

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Only tensors can be divided")

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(np.divide(a.data, b.data), requires_grad=requires_grad, \
                                                      is_leaf=not requires_grad)
        return c     

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = np.divide(grad_output.data, b.data)
        grad_b = - a.data * np.divide(np.ones_like(b.data), np.power(b.data, 2)) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("ReLU can only be applied to tensors")

        ctx.save_for_backward(a)

        c = tensor.Tensor(np.clip(a.data, 0, None), requires_grad=a.requires_grad, \
                                                     is_leaf=not a.requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = np.heaviside(a.data, 0)

        return tensor.Tensor(grad_a), None

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
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

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
    return tensor.Tensor(a, requires_grad = True)

