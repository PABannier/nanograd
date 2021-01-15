import numpy as np
import pyopencl as cl


class GPUBuffer:
    def __init__(self, ctx, shape, hostbuf=None):
        self.shape, self.dtype = tuple(shape), np.float32
        self.cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape), 
                            hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)
    def __repr__(self):
        return f'<GPUBuffer with shape {self.shape}>'

def get_gpu_context_and_queue():
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
        devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)
    return cl_ctx, cl_queue

def buffer_new(ctx, shape, zero=False):
    return GPUBuffer(ctx, shape, hostbuf=None if not zero else np.zeros(shape, dtype=np.float32))

def buffer_np(ctx, x):
    return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def reduce_op(ctx, code, code2, inp, axis=None, start="0.0"):
    if axis is None:
        # full reduce
        osize = [1]*len(inp.shape)
    else:
        osize = np.array(inp.shape)
        osize[list(axis)] = 1
    ret = buffer_new(ctx, osize)
    if axis is None:
        ret.shape = (1,)

    # TODO: this is insanely slow
    prgm = cl.Program(ctx, """
    __kernel void reduce(__global const float *a_g, int sz, __global float *res_g, int prod, int n_dims,
                        __global const int *shape_x, __global const int *shape_ret) {
        int gid = get_global_id(0);
        float out = """+start+""";
        for (int x = 0; x < sz; x++) {
        int idx = 0;  // compute index into a_g
        int tprod = prod;
        int tsz = sz;
        for (int dim = 0; dim < n_dims; dim++) {
            idx *= shape_x[dim];
            if (shape_x[dim] == shape_ret[dim]) {   // dim from gid, don't reduce
            tprod /= shape_x[dim];
            idx += (gid / tprod) % shape_x[dim];
            } else {  // dim from x
            tsz /= shape_x[dim];
            idx += (x / tsz) % shape_x[dim];
            }
        }
        float a = a_g[idx];
        """+code+""";
        }
        res_g[gid] = """+code2+""";
    }""").build()

    prgm.reduce(queue, [np.prod(osize)], None, inp.cl,
        np.int32(np.prod(inp.shape)//np.prod(osize)), ret.cl,
        np.int32(np.prod(osize)), np.int32(len(osize)),
        buffer_np(ctx, np.array(inp.shape, dtype=np.int32)),
        buffer_np(ctx, np.array(osize, dtype=np.int32)))

    return ret

if __name__ == "__main__":
    in_shape = (30, 10, 10)
    a = np.random.normal(0, 1, size=in_shape)
    b = np.empty(1, dtype=np.float32)

    ctx, queue = get_gpu_context_and_queue()
    a_buf = GPUBuffer(ctx, in_shape, hostbuf=a)
    b_buf = reduce_op(ctx, 'out += a', 'out', a_buf)

    cl.enqueue_copy(queue, b, b_buf.cl, is_blocking=True)

    print(b)
    print(np.sum(a))