# Inspired from: https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_gpu.py
import numpy as np
import pyopencl as cl

# *************************************
# ****** OpenCL helper functions ******
# *************************************

class GPUBuffer:
    def __init__(self, ctx, shape, hostbuf=None):
        self.shape, self.dtype = tuple(shape), np.float32
        self.cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape), 
                            hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)
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

def get_binary_op_kernel(code, complist):
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

    return """__kernel void binop(__global const float *x_g, __global const float *y_g, __global float *res_g"""+args+""") {
    int gid0 = get_global_id(0);"""+"".join(compute_idx_rets)+"""
    float a = x_g["""+idx_exprs[0]+"""];
    float b = y_g["""+idx_exprs[1]+"""];
    res_g[gid0] = """+code+""";\n}"""

def get_unary_op_kernel(code):
    return """__kernel void unop(__global const float *x_g, __global float *res_g) {
        int gid = get_global_id(0);
        float a = x_g[gid];
        res_g[gid] = """+code+""";
    }"""

def element_wise_binary_op(ctx, queue, code, x, y):
    n_dims, ret_shape, x_shape, y_shape = compute_output_size(x, y)
    dimension_list, comp_list = compute_broadcasted_dimensions(n_dims, x_shape, y_shape)

    kernel = get_binary_op_kernel(code, tuple(comp_list))
    prgm = cl.Program(ctx, kernel).build()

    out = GPUBuffer(ctx, ret_shape, hostbuf=np.zeros(ret_shape, dtype=np.float32))

    prod_list = np.array(dimension_list, dtype=np.int32)[-1::-1].cumprod(dtype=np.int32)[-1::-1]
    prgm.binop(queue, [prod_list[0]] if len(dimension_list) > 0 else [1], None, 
               x.cl, y.cl, out.cl, *dimension_list, *(prod_list[1:]))
    return out

def unary_op(ctx, queue, code, x):
    out = GPUBuffer(ctx, x.shape, hostbuf=np.zeros(x.shape, dtype=np.float32))
    kernel = get_unary_op_kernel(code)
    prgm = cl.Program(ctx, kernel).build()
    prgm.unop(queue, [np.prod(out.shape)], None, x.cl, out.cl)
    return out

# *************************************
# *********** Forward passes **********
# *************************************

def add_forward(ctx, queue, a, b):
    return element_wise_binary_op(ctx, queue, 'a+b', a, b)

def mul_forward(ctx, queue, a, b):
    return element_wise_binary_op(ctx, queue, 'a*b', a, b)

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