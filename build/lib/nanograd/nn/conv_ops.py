import numpy as np


def get_conv1d_output_size(input_length:int, 
                           kernel_size:int, 
                           stride:int, 
                           padding:int) -> int:
    """Computes the output size of a 1d convolution.

    Args:
        input_length (int): Length of the sequence
        kernel_size (int): Spatial size of the kernel
        stride (int): Stride of the convolution
        padding (int): Spacing between kernel elements

    Returns:
        int: Output size
    """
    input_length += 2 * padding
    return int((input_length - kernel_size) // stride + 1)

def get_conv2d_output_size(input_height:int, input_width:int, 
                           kernel_size:int, stride:int, 
                           padding:int) -> tuple:
    """Computs the output size of a 2d convolution.

    Args:
        input_height (int): Height of the input tensor
        input_width (int): Width of the input tensor
        kernel_size (int): Square size of the kernel
        stride (int): Stride of te convolution
        padding (int): zero-padding added to both sides of the input

    Returns:
        tuple: output height and output width
    """
    input_height += 2 * padding
    input_width += 2 * padding

    output_height = (input_height - kernel_size[0]) // stride + 1
    output_width = (input_width - kernel_size[1]) // stride + 1
    return int(output_height), int(output_width)

def get_im2col_indices(x_shape:tuple, field_height:int, 
                       field_width:int, pad:int, stride:int) -> tuple:
    """Performs a im2col transformation for fast backward passes in 2d
       convolution layer

    Args:
        x_shape (int): Shape of the input tensor
        field_height (int): Height of the kernel
        field_width (int): Width of the kernel
        pad (int): Amount of zero-padding applied in the convolution
        stride (int): Stride of the convolution

    Returns:
        tuple: indices for im2col
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

def col2im(cols:np.ndarray, x_shape:tuple, field_height:int, 
           field_width:int, padding:int, stride:int) -> np.ndarray:
    """Performs a col2im transformation for fast backward passes in 2d


    Args:
        cols (np.ndarray): Input array transformed using im2col
        x_shape (tuple): Shape of the input
        field_height (int): Height of the kernel
        field_width (int): Width of the kernel
        padding (int): Amount of zero-padding applied in the convolution
        stride (int): Stride of the convolution

    Returns:
        np.ndarray: image transformed
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