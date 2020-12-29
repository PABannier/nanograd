import numpy as np
import gzip


MNIST_PATHS = [
    '../data/train-images-idx3-ubyte.gz',
    '../data/train-labels-idx1-ubyte.gz',
    '../data/t10k-images-idx3-ubyte.gz',
    '../data/t10k-labels-idx1-ubyte.gz'
]


def load_mnist():
    print("Loading data...")
    mnist = []
    for path in MNIST_PATHS:
        with open(path, 'rb') as f:
            dat = f.read()
            arr = np.frombuffer(gzip.decompress(dat), dtype=np.uint8)
            mnist.append(arr)
    
    return tuple(mnist)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_conv1d_output_size(input_length, kernel_size, stride, padding=None):
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


def get_conv2d_output_size(input_height, input_width, kernel_size, stride, padding=None):
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