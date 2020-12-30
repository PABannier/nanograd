import numpy as np


def get_conv1d_output_size(input_length, kernel_size, stride, padding):
    r"""
        Gets the size of a Conv1d output.

        Args:
            input_length (int): Length of the sequence
            kernel_size (int): Size of the kernel
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input
            dilation (int): Spacing between kernel elements

        Returns:
            int: size of the output as an int
    """
    input_length += 2 * padding
    return int((input_length - kernel_size) // stride + 1)


def get_conv2d_output_size(input_height, input_width, kernel_size, stride, padding):
    r"""
        Gets the size of a Conv2d output.

        Args:
            input_height (int): Height of the input to the layer
            input_width (int): Width of the input to the layer
            kernel_size (tuple): Size of the kernel (if int the kernel is square)
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input

        Returns:
            int: size of the output as a tuple
    """
    input_height += 2 * padding
    input_width += 2 * padding

    output_height = (input_height - kernel_size[0]) // stride + 1
    output_width = (input_width - kernel_size[1]) // stride + 1
    return int(output_height), int(output_width)


def get_im2col_indices(x_shape, field_height, field_width, pad, stride):
    r"""
        Args:
            x_shape (tuple): shape of the signal
            field_height (int): height of the field (kernel or pool)
            field_width (int): width of the field (kernel or pool)
            padding (int): padding
            stride (int): stride of the operation
            out_height (int): output height of the operation
            out_width (int): output width of the operation
        
        Returns:
            k (int), i (int), j (int)
    """
    N, C, H, W = x_shape
    assert (H + 2 * pad - field_height) % stride == 0
    assert (W + 2 * pad - field_height) % stride == 0
    out_height, out_width = get_conv2d_output_size(H, W, (field_height, field_width), stride, pad)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col(x, field_height, field_width, padding, stride,):
    r"""
        Performs im2col transform

        Args:
            x (np.ndarray): array
            field_height (int): height of the operation
            field_width (int): width of the operation
            padding (int): padding before the operation
            stride (int): stride of the operator 
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im(cols, x_shape, field_height, field_width, padding, stride):
    r"""
        Performs col2im transform

        Args:
            x (np.ndarray): column matrix
            x_shape (tuple): signal shape
            field_height (int): height of the operation (kernel)
            field_width (int): width of the operation (kernel)
            padding (int): padding before the operation
            stride (int): stride of the operator
    """
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