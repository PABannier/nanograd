import numpy as np
import tensor
from autograd_engine import Function


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    return tensor.Tensor(a, requires_grad=True)


def _get_conv1d_output_size(input_length, kernel_size, stride, padding=None):
    r"""
        Gets the size of a Conv1d output.
        NOTE: Padding has already been taken into account in input_height
        and input_width

        Args:
            input_length (int): Length of the sequence
            kernel_size (int): Size of the kernel
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input
            dilation (int): Spacing between kernel elements

        Returns:
            int: size of the output as an int
    """
    if padding is not None:
        input_length += 2 * padding

    return int((input_length - kernel_size) // stride + 1)


def _get_conv2d_output_size(input_height, input_width, kernel_size, stride, padding=None):
    r"""
        Gets the size of a Conv2d output.
        NOTE: Padding has already been taken into account in input_height
        and input_width, hence None as a default value.

        Args:
            input_height (int): Height of the input to the layer
            input_width (int): Width of the input to the layer
            kernel_size (tuple): Size of the kernel (if int the kernel is square)
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input

        Returns:
            int: size of the output as a tuple
    """
    if padding is not None:
        input_height += 2 * padding
        input_width += 2 * padding 

    output_height = (input_height - kernel_size[0]) // stride + 1
    output_width = (input_width - kernel_size[1]) // stride + 1
    return int(output_height), int(output_width)


def im2col(x, field_height, field_width, padding, stride):
    N, C, H, W = x.shape

    HH = int((H + 2 * padding - field_height) / stride + 1)
    WW = int((W + 2 * padding - field_width) / stride + 1)

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    cols = np.zeros((C * field_height * field_width, N * HH * WW), dtype=x.dtype)

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]
        
    return cols

def col2im_6d(cols, N, C, H, W, HH, WW, pad, stride):
    r"""
        Performs a column matrix to image transform.
        It is the inverse transform of im2col.
        col2im is mainly used for backpropagation of convolutional and pooling
        layers.

        Args:
            cols (np.ndarray): matrix of columns
            N (int): batch size
            C (int): number of input channels 
            H (int): height of the input volume
            W (int): width of the input volume
            HH (int): kernel height
            WW (int): kernel width
            pad (int): padding of the convolution
            stride (int): stride of the convolution
        
        Returns:
            res (np.ndarray): image matrix
    """
    x = np.empty((N, C, H, W), dtype=cols.dtype)

    OH = int((H + 2 * pad - HH) / stride + 1)
    OW = int((W + 2 * pad - WW) / stride + 1)

    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)

    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(OH):
                        for w in range(OW):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]
    
    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    r""" An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def max_pool_2d_forward_reshape(x, kernel_size, stride):
    r"""
        Performs an efficient forward pass of MaxPool2d.
        Only available for square pooling regions.

        Args:
            x (np.ndarray): activation map after a 2d convolution
            kernel_size (tuple): pooling kernel dimensions
            stride (int): stride of the pooling operation
        
        Returns:
            x_reshaped (np.ndarray): reshaped activation map
            out (np.ndarray): pooled activation map
    """
    N, C, H, W = x.shape
    HH, WW = kernel_size

    assert HH == WW == stride, 'Invalid pool params'
    assert H % HH == 0
    assert W % WW == 0

    x_reshaped = x.reshape(N, C, H // HH, HH, W // WW, WW)
    out = x_reshaped.max(axis=3).max(axis=4)

    return x_reshaped, out


def max_pool_2d_backward_reshape(grad_output, x, x_reshaped, out):
    r"""
        Performs an efficient backward pass of MaxPool2d.
        Only available for square pooling regions.

        Args:
            grad_output (np.ndarray): gradient
            x (np.ndarray): activation map
            x_reshaped (np.ndarray): reshaped activation map
            out (np.ndarray): pooled activation map
        
        Returns:
            grad_x (np.ndarray): gradient of the input
    """
    grad_x_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)

    grad_out_newaxis = grad_output[:, :, :, np.newaxis, :, np.newaxis]
    grad_out_broadcast, _ = np.broadcast_arrays(grad_out_newaxis, grad_x_reshaped)
    grad_x_reshaped[mask] = grad_out_broadcast[mask]
    grad_x_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    grad_x = grad_x_reshaped.reshape(x.shape)

    return grad_x


def max_pool_2d_forward_im2col(x, kernel_size, stride):
    r"""
        Performs a forward pass of MaxPool2d, using im2col.
        Main case suited for all pooling region shapes.

        Args: 
            x (np.ndarray): activation map after a 2d convolution
            kernel_size (tuple): pooling kernel dimensions
            stride (int): stride of the pooling operation

        Returns:
            x_reshaped (np.ndarray): reshaped activation map
            out (np.ndarray): pooled activation map
    """
    N, C, H, W = x.shape 
    HH, WW = kernel_size

    assert (H - HH) % stride == 0, 'Invalid height'
    assert (W - WW) % stride == 0, 'Invalid width'

    OH = (H - HH) // stride + 1
    OW = (W - WW) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)

    x_cols = im2col(x_split, HH, WW, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]

    out = x_cols_max.reshape(OH, OW, N, C).transpose(2, 3, 0, 1)

    return x_cols, x_cols_argmax, out


def max_pool_2d_backward_im2col(grad_output, x, x_cols, x_cols_argmax, 
                                pool_height, pool_width, stride):
    r"""
        Performs a backward pass of MaxPool2d, using im2col.
        Main case suited for all pooling region shapes.

        Args:
            grad_output (np.ndarray): gradient
            x (np.ndarray): activation map
            x_cols (np.ndarray): im2col-transformed image
            x_cols_argmax (np.ndarray): pooled im2col-transformed image
            pool_height (int): 
            pool_width (int):
            stride (int): 
        
        Returns:
            grad_x (np.ndarray): gradient of the input

    """
    N, C, H, W = x.shape

    grad_output_reshaped = grad_output.transpose(2, 3, 0, 1).flatten()

    grad_x_cols = np.zeros_like(x_cols)
    grad_x_cols[x_cols_argmax, np.arange(grad_x_cols.shape[1])] = grad_output_reshaped

    grad_x = col2im_indices(grad_x_cols, (N * C, 1, H, W), pool_height, pool_width, \
                            padding=0, stride=stride)
    grad_x = grad_x.reshape(x.shape)

    return grad_x


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
        b = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad, 
                                          is_leaf=not requires_grad)
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
        b = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad, 
                                           is_leaf=not requires_grad)
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

        grad_a = tensor.Tensor(grad_a)

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
    def forward(ctx, x, weight, bias, stride, padding):
        """
            The forward/backward of a Conv1d Layer in the comp graph.
            
            Args:
                x (Tensor): (batch_size, in_channel, input_size) input data
                weight (Tensor): (out_channel, in_channel, kernel_size)
                bias (Tensor): (out_channel,)
                stride (int): Stride of the convolution
                padding (int): Padding for the convolution
            
            Returns:
                Tensor: (batch_size, out_channel, output_size) output data
        """
        N, C, L = x.shape
        F, _, KL = weight.shape

        # Padding the input
        x_padded = np.pad(x.data, (padding, padding), mode="constant") 

        # Saving relevant tensors for backward
        ctx.save_for_backward(x, weight, bias)

        # Defining output shapes
        L += 2 * padding
        OL = _get_conv1d_output_size(L, KL, stride)
        out = np.zeros((N, F, OL))

        # Im2Col operation
        strides = (L, 1, C * L, stride * L)
        strides = x.data.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(
            x=x_padded,
            shape=(C, L, N, OL),
            strides=strides
        )
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * KL, N * OL)

        res = weight.data.reshape(F, -1) @ x_cols + bias.data.reshape(-1, 1)
        res.shape = (F, N, OL)
        out = res.transpose(1, 0, 2)
        
        return tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        raise NotImplementedError("Implement functional.Conv1d.backward()!")


class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        """
            The forward/backward of a Conv2d Layer in the comp graph.
            
            Args:
                x (Tensor): (batch_size, in_channel, input_height, input_width) input data
                weight (Tensor): (out_channel, in_channel, kernel_height, kernel_width)
                bias (Tensor): (out_channel,)
                stride (int): Stride of the convolution
                padding (int): Padding for the convolution
            
            Returns:
                Tensor: (batch_size, out_channel, output_height, output_width) output data
        """
        # Retrieving the dimensions of the input and weight tensors
        N, C, H, W = x.shape
        F, _, HH, WW = weight.shape

        # Padding the input
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

        # Defining output shapes
        H += 2 * padding
        W += 2 * padding
        OH, OW = _get_conv2d_output_size(H, W, (HH, WW), stride)
        out = np.zeros((N, F, OH, OW))

        # Im2Col operation
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.data.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(
            x=x_padded,
            shape=(C, HH, WW, N, OH, OW),
            strides=strides
        )
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * OH * OW)

        # Perform convolution as a matrix multiplication
        res = weight.data.reshape(F, -1) @ x_cols + bias.data.reshape(-1, 1)
        res.shape = (F, N, OH, OW)
        out = res.transpose(1, 0, 2, 3)

        # Saving relevant tensors for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.x_cols = x_cols
        ctx.stride, ctx.pad = stride, padding
        
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

        tmp_grad = grad_output.data.transpose(1, 0, 2, 3).reshape(F, -1)
        grad_weight = (tmp_grad @ x_cols.T).reshape(weight.shape)
        
        grad_x_cols = weight.data.reshape(F, -1).T @ tmp_grad
        grad_x_cols.shape = (C, HH, WW, N, OH, OW)
        grad_x = col2im_6d(grad_x_cols, N, C, H, W, HH, WW, pad, stride)

        grad_x = tensor.Tensor(grad_x)
        grad_weight = tensor.Tensor(grad_weight)
        grad_bias = tensor.Tensor(grad_bias)

        return grad_x, grad_weight, grad_bias


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        N, C, H, W = x.shape
        HH, WW = kernel_size

        same_size = HH == WW == stride
        tiles = (H % HH == 0) and (W  % WW == 0)

        if same_size and tiles:
            x_reshaped, res = max_pool_2d_forward_reshape(x.data, kernel_size, stride)
            ctx.method = 'reshape'
            ctx.x_reshaped = x_reshaped
        else:
            x_cols, x_cols_argmax, res = max_pool_2d_forward_im2col(x.data, kernel_size, stride)
            ctx.method = 'im2col'
            ctx.x_cols = x_cols
            ctx.x_cols_argmax = x_cols_argmax
        
        res = tensor.Tensor(res, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
        ctx.save_for_backward(x, res)
        ctx.kernel_size, ctx.stride = kernel_size, stride

        return res   

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        kernel_size, stride = ctx.kernel_size, ctx.stride
        method = ctx.method

        if method == 'reshape':
            x_reshaped = ctx.x_reshaped 
            grad = max_pool_2d_backward_reshape(grad_output.data, x.data, x_reshaped, out.data)
        elif method == 'im2col':
            x_cols, x_cols_argmax = ctx.x_cols, ctx.x_cols_argmax
            grad = max_pool_2d_backward_im2col(grad_output.data, x.data, x_cols, x_cols_argmax, \
                                               kernel_size[0], kernel_size[1], stride)
        else:
            raise Exception(f'Methods available for MaxPool2d backward: "reshape" and "im2col". Got {method}.')

        return tensor.Tensor(grad),
