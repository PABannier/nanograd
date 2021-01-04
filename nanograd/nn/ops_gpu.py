import numpy as np
import pyopencl as cl
from nanograd.tensor import GPUBuffer, Device, Tensor

def create_buffer(ctx, shape, zero=False):
    return GPUBuffer(shape, hostbuf=None if not zero else np.zeros(shape))

def buffer_np(ctx, x):
  return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)