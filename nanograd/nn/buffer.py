import numpy as np
import warnings 

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL is not available on this computer. Can't use \
                   parallel computing. Please install it to move comptutations \
                   to move the GPU.")


class GPUBuffer:
    def __init__(self, ctx, shape, hostbuf=None):
        if isinstance(shape, int): shape = [shape]
        self.shape, self.dtype = tuple(shape), np.float32

        if isinstance(hostbuf, GPUBuffer):
            self.cl = hostbuf.cl
        else:
            self.cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 
                                np.int32(4*np.prod(shape)), hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)

    def __repr__(self):
        return f'<GPUBuffer with shape {self.shape}>'