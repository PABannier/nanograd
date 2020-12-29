import numpy as np


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