# Inspired from: https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_gpu.py
import numpy as np
import pyopencl as cl
import functools

# *************************************
# ****** OpenCL helper functions ******
# *************************************

class GPUBuffer:
    def __init__(self, ctx, shape, hostbuf=None):
        if isinstance(shape, int): shape = [shape]
        self.shape, self.dtype = tuple(shape), np.float32

        if isinstance(hostbuf, GPUBuffer):
            self.cl = hostbuf.cl
        else:
            self.cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 
                                4*np.prod(shape), hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)

    def __repr__(self):
        return f'<GPUBuffer with shape {self.shape}>'

def compute_output_size(x, y):
    n_dims = max(len(x.shape), len(y.shape))
    x_shape, y_shape = np.ones(n_dims, dtype=np.int32), np.ones(n_dims, dtype=np.int32)
    x_shape[:len(x.shape)] = np.array(x.shape, dtype=np.int32)
    y_shape[:len(y.shape)] = np.array(y.shape, dtype=np.int32)

    if not np.all((x_shape == 1) | (y_shape == 1) | (x_shape == y_shape)):
        raise Exception(f"Unbroadcastable shapes: {x.shape} vs {y.shape}")

    ret_shape = np.maximum(x_shape, y_shape)
    return n_dims, ret_shape, x_shape, y_shape

def compute_broadcasted_dimensions(n_dims, x_shape, y_shape):
    dimension_list, comp_list = [], []

    def push(dim, comp):
        if len(comp_list) > 0 and comp_list[-1] == comp:
            dimension_list[-1] *= dim
        elif comp != (False, False):
            dimension_list.append(dim)
            comp_list.append(comp)
    
    for i in range(n_dims):
        push(np.int32(max(x_shape[i], y_shape[i])), 
             (x_shape[i] > 1, y_shape[i] > 1))
    return dimension_list, comp_list

@functools.lru_cache()
def get_binary_op_kernel(ctx, code, complist):
    ndims = len(complist)
    # Arguments for the kernel function. E.g.: int d0, int d1, int d2, int p0, int p1
    # E.g. (cont'd): d0, d1, d2 are the dimensions of the tensor
    # E.g. (cont'd): p0, p1 are the cumulative dimensions in reverse order of the tensor
    args = "".join([f", int d{i}" for i in range(ndims)]) + "".join([f", int p{i}" for i in range(ndims-1)])

    # Computes the indices of the element in the tensor
    # Remember that OpenCL flattened the multi-dimensional tensor into a 1d vector
    compute_idx_rets = ["\n    int idx_ret" + str(i) + " = (gid0 / " +
                        (f"p{i}" if i < ndims - 1 else "1") + ") % d" + 
                        str(i) + ";" for i in range(ndims)]

    # Computes the actual index in the flattened OpenCL tensor
    # Recursive formula: idx_ret2 + d2*(idx_ret1 + d1*(idx_ret0 + d0*(0)))
    idx_exprs = ["0", "0"] # [idx_x, idx_y]
    for i in range(ndims):
        for j in range(2):
            if complist[i][j]:
                idx_exprs[j] = "idx_ret%d + d%d*(%s)" % (i, i, idx_exprs[j])

    return cl.Program(ctx, """__kernel void binop(__global const float *x_g, __global const float *y_g, __global float *res_g"""+args+""") {
    int gid0 = get_global_id(0);"""+"".join(compute_idx_rets)+"""
    float a = x_g["""+idx_exprs[0]+"""];
    float b = y_g["""+idx_exprs[1]+"""];
    res_g[gid0] = """+code+""";\n}""").build()

@functools.lru_cache()
def get_unary_op_kernel(ctx, code):
    return cl.Program(ctx, """__kernel void unop(__global const float *x_g, __global float *res_g) {
        int gid = get_global_id(0);
        float a = x_g[gid];
        res_g[gid] = """+code+""";
    }""").build()

@functools.lru_cache()
def get_reduce_op_kernel(ctx, code, code2, start):
    return cl.Program(ctx, """
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

def element_wise_binary_op(ctx, queue, code, x, y):
    n_dims, ret_shape, x_shape, y_shape = compute_output_size(x, y)
    dimension_list, comp_list = compute_broadcasted_dimensions(n_dims, x_shape, y_shape)

    prgm = get_binary_op_kernel(ctx, code, tuple(comp_list))

    out = GPUBuffer(ctx, ret_shape, hostbuf=np.zeros(ret_shape, dtype=np.float32))

    prod_list = np.array(dimension_list, dtype=np.int32)[-1::-1].cumprod(dtype=np.int32)[-1::-1]
    prgm.binop(queue, [prod_list[0]] if len(dimension_list) > 0 else [1], None, 
               x.cl, y.cl, out.cl, *dimension_list, *(prod_list[1:]))
    return out

def unary_op(ctx, queue, code, x):
    out = GPUBuffer(ctx, x.shape, hostbuf=np.zeros(x.shape, dtype=np.float32))
    prgm = get_unary_op_kernel(ctx, code)
    prgm.unop(queue, [np.prod(out.shape)], None, x.cl, out.cl)
    return out

def reduce_op(ctx, queue, code, code2, inp, axis=None, keepdims=False, start="0.0"):
    if isinstance(axis, int): axis = [axis]
    if axis is None:
        osize = [1]*len(inp.shape)
    else:
        osize = np.array(inp.shape)
        osize[list(axis)] = 1

    ret = GPUBuffer(ctx, osize if keepdims else osize[osize != 1], hostbuf=None)

    if axis is None: 
        ret.shape = (1,)
    
    prgm = get_reduce_op_kernel(ctx, code, code2, start)

    prgm.reduce(queue, [np.prod(osize)], None, inp.cl,
        np.int32(np.prod(inp.shape) // np.prod(osize)), ret.cl,
        np.int32(np.prod(osize)), np.int32(len(osize)),
        cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                  hostbuf=np.array(inp.shape, dtype=np.int32)),
        cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                  hostbuf=np.array(osize, dtype=np.int32))
    )
    return ret

def perm_axis_op(ctx, queue, inp, order=(1,0)):
    out_size = np.array(inp.shape)[list(order)]
    ret = GPUBuffer(ctx, out_size)
    prgm = cl.Program(ctx, """
        __kernel void perm(__global const float *a_g, __global float *res_g, int n_axis,
                            __global const int *shape, __global const int *order) {
            int gid = get_global_id(0);
            int gi = gid;
            int idx = 0;
            for(int i = n_axis-1; i>-1; i--) {
            int stride = 1;
            for(int j=order[i]+1; j<n_axis; j++) stride *= shape[j];
            idx += (gi % shape[order[i]])*stride;
            gi /= shape[order[i]];
            }
            res_g[gid] = a_g[idx];
            }"""
    ).build()

    prgm.perm(queue, [np.prod(out_size)], None, inp.cl, ret.cl, np.int32(len(out_size)),
         cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(inp.shape, dtype=np.int32)),
         cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(order, dtype=np.int32)))
    return ret

def inner_slice(ctx, queue, inp, arg):
    shift = [y[0] for y in arg]
    out_shape = [y[1]-y[0] for y in arg]
    ret = GPUBuffer(ctx, out_shape)
    prgm = cl.Program(ctx, """
        __kernel void gslice(__global const float *input, __global float *output, int prod, int n_dims,
                       __global const int *shape_x, __global const int *shape_ret,
                       __global const int *shift) {
            int gid = get_global_id(0);
            int iptr = 0;
            int zero = 1;
            for (int dim = 0; dim < n_dims; dim++) {
            prod /= shape_ret[dim];
            int sidx = (gid / prod) % shape_ret[dim] + shift[dim];
            zero &= (sidx >= 0 && sidx < shape_x[dim]);
            iptr = (iptr * shape_x[dim]) + sidx;
            }
            output[gid] = zero ? input[iptr] : 0.0;
        }
    """).build()

    prgm.gslice(queue, [np.prod(ret.shape)], None, inp.cl, ret.cl, np.int32(np.prod(ret.shape)), np.int32(len(ret.shape)),
                cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(inp.shape, dtype=np.int32)),
                cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(ret.shape, dtype=np.int32)),
                cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(shift, dtype=np.int32)))
    return ret

def matmul_op(ctx, queue, a, b):
    cnt = np.prod(a.shape[0:-2]) if len(a.shape) > 2 else 1
    isize, msize, osize = np.int32(a.shape[-2]), np.int32(a.shape[-1]), np.int32(b.shape[-1])
    ret = GPUBuffer(ctx, list(a.shape[0:-2])+[isize, osize])
    prgm = cl.Program(ctx, """
        __kernel void matmul(__global const float *input, __global const float *weight, __global float *res,
                            int isize, int is0, int is1, int msize, int ws0, int ws1, int osize) {
            int stride = get_global_id(2);
            int X = get_global_id(0); // isize
            int Y = get_global_id(1); // osize
            float ret = 0.0;
            for (int x = 0; x < msize; x++) {
                ret += input[X * is0 + x * is1 + isize*msize*stride] *
                weight[Y * ws0 + x * ws1 + msize*osize*stride];
            }
            res[X * osize + Y + isize*osize*stride] = ret;
        }
    """).build()

    prgm.matmul(queue, [isize, osize, cnt], None, 
                a.cl, b.cl, ret.cl, isize, msize, 
                np.int32(1), msize, np.int32(1), osize, osize)
    return ret


# *************************************
# *********** Forward passes **********
# *************************************

def add_forward(ctx, queue, a, b):
    return element_wise_binary_op(ctx, queue, 'a+b', a, b)

def mul_forward(ctx, queue, a, b):
    return element_wise_binary_op(ctx, queue, 'a*b', a, b)

def matmul_forward(ctx, queue, a, b):
    return matmul_op(ctx, queue, a, b)

def log_forward(ctx, queue, a):
    return unary_op(ctx, queue, 'log(a)', a)

def exp_forward(ctx, queue, a):
    return unary_op(ctx, queue, 'exp(a)', a)

def neg_forward(ctx, queue, a):
    return unary_op(ctx, queue, '-a', a)

def pow_forward(ctx, queue, a, exp):
    return unary_op(ctx, queue, f'pow(a, (float){exp})', a)

def relu_forward(ctx, queue, a):
    return unary_op(ctx, queue, 'max(a, (float)0.)', a)

def sigmoid_forward(ctx, queue, a):
    return unary_op(ctx, queue, '1.0 / (1 + exp(-a))', a)

def tanh_forward(ctx, queue, a):
    return unary_op(ctx, queue, '(exp(a) - exp(-a)) / (exp(a) + exp(-a))', a)

def slice_forward(ctx, queue, a, indices):
    return inner_slice(ctx, queue, a, indices)

def transpose_forward(ctx, queue, a):
    return perm_axis_op(ctx, queue, a, order=(1, 0))

def reshape_forward(ctx, queue, a, shape):
    new_shape = tuple([-np.prod(a.shape) // np.prod(shape) 
                       if s == -1 else s for s in shape])
    assert np.prod(new_shape) == np.prod(shape), "Inconsistent array reshape size"
    return GPUBuffer(ctx, new_shape, hostbuf=a.data)

def max_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out = max(a, out)', 'out',
                     a, axis=axis, keepdims=keepdims, start='-INFINITY')

def min_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out = min(a, out)', 'out',
                     a, axis=axis, keepdims=keepdims, start='+INFINITY')

def sum_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out += a', 'out', 
                     a, axis=axis, keepdims=keepdims)

def conv1d_forward(ctx, queue, a, weight, bias, stride, pad):
    raise NotImplementedError

def conv2d_forward(ctx, queue, a, weight, bias, stride, pad):
    raise NotImplementedError
