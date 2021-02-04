# Inspired from: https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_gpu.py
import numpy as np
import functools

from nanograd.nn.conv_ops import get_conv1d_output_size, get_conv2d_output_size

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True

except ImportError:
    PYOPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL is not available on this computer. Can't use \
                   parallel computing. Please install it to move comptutations \
                   to move the GPU.")

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
                                np.int32(4*np.prod(shape)), hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)

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

def unbroadcast(ctx, queue, out, in_shape):
    sum_axis = [i for i in range(len(in_shape)) if in_shape[i]==1 and out.shape[i]>1] if in_shape != (1,) else None
    return reduce_op(ctx, queue, "out += a", "out", out, sum_axis, keepdims=True)

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

def one_hot_encoding(ctx, queue, a, num_classes):
    ret = GPUBuffer(ctx, (a.shape[0], num_classes))
    n_rows, n_cols = np.int32(a.shape[0]), np.int32(num_classes)
    prgm = cl.Program(ctx, """
        __kernel void one_hot(__global const float *labels, __global float *res, 
                              const int n_cols) {
            int gid0 = get_global_id(0);

            int label = labels[gid0];
            res[gid0 * n_cols + label] = 1;
        }
    """).build()

    prgm.one_hot(queue, [n_rows, n_cols], None, a.cl, ret.cl, n_cols)
    return ret

def pad1d_op(ctx, queue, a, pad, length):
    padded_length = np.int32(length + 2 * pad)
    batch_size, in_channel = np.int32(a.shape[0]), np.int32(a.shape[1])
    ret = GPUBuffer(ctx, (batch_size, in_channel, padded_length))
    prgm = cl.Program(ctx, """
        __kernel void pad1d(__global const float *a, __global float *res, const int batch_size,
                            const int in_channel, const int padded_length, const int length,
                            const int pad) {
            
            int batch_id = get_global_id(0);
            int channel_id = get_global_id(1);
            int col = get_global_id(2);

            if((col - pad >= 0) && (col - pad < length)) {
                res[batch_id * in_channel * padded_length + channel_id * padded_length + col] = \
                  a[batch_id * in_channel * length + channel_id * length + col-pad];
            }
        }
    """).build()

    prgm.pad1d(queue, [batch_size, in_channel, padded_length], None, a.cl, ret.cl,
               batch_size, in_channel, padded_length, np.int32(length), np.int32(pad))
    return ret

def pad2d_op(ctx, queue, a, pad, im_height, im_width):
    padded_height, padded_width = np.int32(im_height + 2 * pad), np.int32(im_width + 2 * pad)
    batch_size, in_channel = np.int32(a.shape[0]), np.int32(a.shape[1])
    ret = GPUBuffer(ctx, (batch_size, in_channel, padded_height, padded_width))
    prgm = cl.Program(ctx, """
        __kernel void pad2d(__global const float *a, __global float *res, const int batch_size, 
                            const int in_channel, const int padded_height, const int padded_width,
                            const int im_height, const int im_width, const int pad) {

            int batch_id = get_global_id(0) / in_channel;
            int channel_id = get_global_id(0) % in_channel;
            int row = get_global_id(1);
            int col = get_global_id(2);

            if((row - pad >= 0) && (col - pad >= 0) && (col - pad < im_width) && (row - pad < im_height)) {
                res[batch_id * in_channel * padded_height * padded_width + channel_id * padded_height * padded_width + row * padded_width + col] = \
                  a[batch_id * in_channel * im_height * im_width + channel_id * im_height * im_width + (row - pad) * im_width + col-pad];
            }
        }
    """).build()
    
    prgm.pad2d(queue, [batch_size * in_channel, padded_height, padded_width], 
               None, a.cl, ret.cl, batch_size, in_channel, padded_height, padded_width,
               np.int32(im_height), np.int32(im_width), np.int32(pad))
    return ret

def conv1d_op(ctx, queue, a, weight, stride, output_length):
    batch_size, stride = np.int32(a.shape[0]), np.int32(stride)
    output_length = np.int32(output_length)
    num_filters, in_channel = np.int32(weight.shape[0]), np.int32(weight.shape[1])
    kernel_length = np.int32(weight.shape[2])
    length = np.int32(a.shape[2])

    ret = GPUBuffer(ctx, (batch_size, num_filters, output_length))
    prgm = cl.Program(ctx, """
        __kernel void conv1d(__global const float *input, __global const float *weight,
                             __global float *output, const int kernel_length, const int num_filters, const int in_channel,
                             const int out_length, const int length, const int stride) {
            int batch_id = get_global_id(0);
            int filter_id = get_global_id(1);
            int col = get_global_id(2);

            float res = 0.0;

            for(int c = 0; c < in_channel; c++) {
                for (int x = col * stride; x < col * stride + kernel_length; x++) {
                    res += input[batch_id * in_channel * length + c * length + x] * \
                           weight[filter_id * in_channel * kernel_length + c * kernel_length + \
                                  (x - col * stride)];
                }
            }

            output[batch_id * num_filters * out_length + filter_id * out_length + col] = res;
        }
    """).build()

    prgm.conv1d(queue, [batch_size, num_filters, output_length], None, a.cl, weight.cl, 
                ret.cl, kernel_length, num_filters, in_channel, np.int32(output_length), length, 
                stride)
    return ret

def conv2d_op(ctx, queue, a, weight, stride, output_height, output_width):
    batch_size, stride = np.int32(a.shape[0]), np.int32(stride)
    output_height, output_width = np.int32(output_height), np.int32(output_width)
    num_filters, in_channel = np.int32(weight.shape[0]), np.int32(weight.shape[1])
    kernel_height, kernel_width  = np.int32(weight.shape[2]), np.int32(weight.shape[3])
    im_height, im_width = np.int32(a.shape[2]), np.int32(a.shape[3])

    ret = GPUBuffer(ctx, (batch_size, num_filters, output_height, output_width))
    prgm = cl.Program(ctx, """
        __kernel void conv2d(__global const float *input, __global const float *weight,
                             __global float *output, const int kernel_height, const int kernel_width, const int num_filters, 
                             const int in_channel, const int out_height, const int out_width, const int im_height, 
                             const int im_width, const int stride) {
            int gid0 = get_global_id(0);
            int row_id = get_global_id(1);
            int col_id = get_global_id(2);

            int batch_id = gid0 / num_filters;
            int filter_id = gid0 % num_filters;

            float res = 0.0;

            for(int c = 0; c < in_channel; c++) {
                for(int y = row_id * stride; y < row_id * stride + kernel_height; y++) {
                    for(int x = col_id * stride; x < col_id * stride + kernel_width; x++) {
                        res += input[batch_id*in_channel*im_height*im_width + c*im_height*im_width + y*im_width + x] * \
                               weight[filter_id*in_channel*kernel_height*kernel_width + c*kernel_height*kernel_width + \
                                      (y-row_id*stride)*kernel_width + (x-col_id*stride)];
                    }
                }
            }

           output[batch_id * num_filters * out_height * out_width + filter_id * out_height * out_width + row_id * out_width + col_id] = res; 
        }
    """).build()

    prgm.conv2d(queue, [batch_size * num_filters, output_height, output_width], None, a.cl, weight.cl,
                ret.cl, kernel_height, kernel_width, num_filters, in_channel, output_height, output_width, im_height,
                im_width, stride)
    return ret


# *************************************
# *********** Forward passes **********
# *************************************

def squeeze_forward(ctx, queue, a, axis):
    in_shape = np.array(a.shape)
    out_shape = np.delete(in_shape, axis).astype(np.int32)
    return GPUBuffer(ctx, out_shape, hostbuf=a.data)

def unsqueeze_forward(ctx, queue, a, axis):
    in_shape = np.array(a.shape)
    out_shape = np.insert(in_shape, axis, 1).astype(np.int32)
    return GPUBuffer(ctx, out_shape, hostbuf=a.data)
    
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
    shape = tuple(-np.prod(a.shape) // np.prod(shape) if s == -1 else s for s in shape)
    return GPUBuffer(ctx, shape, hostbuf=a.data)

def max_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out = max(a, out)', 'out',
                     a, axis=axis, keepdims=keepdims, start='-INFINITY')

def min_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out = min(a, out)', 'out',
                     a, axis=axis, keepdims=keepdims, start='+INFINITY')

def sum_forward(ctx, queue, a, axis, keepdims):
    return reduce_op(ctx, queue, 'out += a', 'out', 
                     a, axis=axis, keepdims=keepdims)

def conv1d_forward(ctx, queue, a, weight, stride):
    batch_size, in_channel, length = a.shape
    num_filters, _, kernel_length = weight.shape
    output_length = get_conv1d_output_size(length, kernel_length, stride, 0)

    out = conv1d_op(ctx, queue, a, weight, stride, output_length)
    return out

def conv2d_forward(ctx, queue, a, weight, stride):
    batch_size, in_channel, im_height, im_width = a.shape
    num_filters, _, kernel_height, kernel_width = weight.shape

    output_height, output_width = get_conv2d_output_size(im_height, im_width,
        (kernel_height, kernel_width), stride, 0)

    out = conv2d_op(ctx, queue, a, weight, stride, output_height, output_width)
    return out

# *************************************
# ********** Backward passes **********
# *************************************

def squeeze_backward(ctx, queue, grad_output, axis):
    in_shape = np.array(grad_output.shape)
    out_shape = np.insert(in_shape, axis, 1).astype(np.int32)
    return GPUBuffer(ctx, out_shape, hostbuf=grad_output.data)

def unsqueeze_backward(ctx, queue, grad_output, axis):
    in_shape = np.array(grad_output.shape)
    out_shape = np.delete(in_shape, axis).astype(np.int32)
    return GPUBuffer(ctx, out_shape, hostbuf=grad_output.data)

def add_backward(ctx, queue, grad_output, a_shape, b_shape):
    return unbroadcast(ctx, queue, grad_output, a_shape), unbroadcast(ctx, queue, grad_output, b_shape) 

def mul_backward(ctx, queue, grad_output, a, b):
    grad_a = element_wise_binary_op(ctx, queue, 'a*b', grad_output, b)
    grad_b = element_wise_binary_op(ctx, queue, 'a*b', grad_output, a)
    return unbroadcast(ctx, queue, grad_a, a.shape), unbroadcast(ctx, queue, grad_b, b.shape)

def matmul_backward(ctx, queue, grad_output, a, b):
    aT = perm_axis_op(ctx, queue, a)
    bT = perm_axis_op(ctx, queue, b)
    grad_a = matmul_op(ctx, queue, grad_output, bT)
    grad_b = matmul_op(ctx, queue, aT, grad_output)
    return grad_a, grad_b

def log_backward(ctx, queue, grad_output, a):
    return element_wise_binary_op(ctx, queue, 'a/b', grad_output, a)

def exp_backward(ctx, queue, grad_output, a):
    return element_wise_binary_op(ctx, queue, 'a * exp(b)', grad_output, a)

def neg_backward(ctx, queue, grad_output):
    return unary_op(ctx, queue, '-a', grad_output)

def pow_backward(ctx, queue, grad_output, a, exp):
    return element_wise_binary_op(ctx, queue, f'a * {exp} * pow(b, (float){exp-1})', grad_output, a)

def relu_backward(ctx, queue, grad_output, a):
    return element_wise_binary_op(ctx, queue, 'a*(b>=0)', grad_output, a)

def sigmoid_backward(ctx, queue, grad_output, a):
    return element_wise_binary_op(ctx, queue, 'a*(exp(-b) / pow((1 + exp(-b)), 2))', 
                                  grad_output, a)

def tanh_backward(ctx, queue, grad_output, a):
    return element_wise_binary_op(ctx, queue, 'a*(1 - pow(exp(b) - exp(-b), 2) / pow(exp(b) + exp(-b), 2))', 
                                  grad_output, a)

def slice_backward(ctx, queue, grad_output, shape, fwd_indices):
    indices = [(0 - p[0], grad_output.shape[i] + (shape[i] - p[1])) for i, p in enumerate(fwd_indices)]
    return inner_slice(ctx, queue, grad_output, indices)

def transpose_backward(ctx, queue, grad_output):
    return perm_axis_op(ctx, queue, grad_output)

def reshape_backward(ctx, queue, grad_output, shape):
    new_shape = tuple([-np.prod(grad_output.shape) // np.prod(shape) 
                       if s == -1 else s for s in shape])
    assert np.prod(new_shape) == np.prod(shape), "Inconsistent array reshape size"
    return GPUBuffer(ctx, new_shape, hostbuf=grad_output)

def max_backward(ctx, queue, grad_output, inp, out, axis):
    shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
    ret2 = element_wise_binary_op(ctx, queue, "1.0*(a==b)", inp, GPUBuffer(ctx, shape, out))
    div = reduce_op(ctx, queue, "out += a", "out+1e-10", ret2, axis=axis)
    ret3 = element_wise_binary_op(ctx, queue, "a/b", ret2, GPUBuffer(ctx, shape, div))
    return element_wise_binary_op(ctx, queue, 'a*b', ret3, GPUBuffer(ctx, shape, grad_output))

def min_backward(ctx, queue, grad_output, inp, out, axis):
    shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
    ret2 = element_wise_binary_op(ctx, queue, "1.0*(a==b)", inp, GPUBuffer(ctx, shape, out))
    div = reduce_op(ctx, queue, "out += a", "out+1e-10", ret2, axis=axis)
    ret3 = element_wise_binary_op(ctx, queue, "a/b", ret2, GPUBuffer(ctx, shape, div))
    return element_wise_binary_op(ctx, queue, 'a*b', ret3, GPUBuffer(ctx, shape, grad_output))

def sum_backward(ctx, queue, grad_output, a, axis):
    axis = [axis] if type(axis) == int else axis
    shape = [1 if axis is None or i in axis else a.shape[i] for i in range(len(a.shape))]
    output = GPUBuffer(ctx, shape, hostbuf=grad_output)
    return element_wise_binary_op(ctx, queue, 'a+b', output, GPUBuffer(ctx, a.shape))

def conv1d_backward(ctx, queue, grad_output, x, weight, stride):
    batch_size, _, out_length = grad_output.shape
    num_filters, _, kernel_length = weight.shape
    _, in_channel, in_length = x.shape

    grad_x = GPUBuffer(ctx, shape=x.shape)
    grad_weight = GPUBuffer(ctx, shape=weight.shape)

    prgm_grad_x = cl.Program(ctx, """
        __kernel void conv_backward_x(__global const float *weight, __global const float *grad_output, __global float *grad_x, const int kernel_length,
                                      const int num_filters, const int in_channel, const int out_length, const int in_length, const int stride,
                                      const int batch_size) {

            int batch = get_global_id(0);
            int channel = get_global_id(1);

            for(int x = 0; x < out_length; x++) {
                for(int kx = 0; kx < kernel_length; kx++) {

                    float sum = 0.0;

                    for(int f = 0; f < num_filters; f++) {
                        sum += grad_output[batch * num_filters * out_length + f * out_length + x] * \
                            weight[f * in_channel * kernel_length + channel * kernel_length + kx];
                    } 

                    grad_x[batch * in_channel * in_length + channel * in_length + x * stride + kx] += sum;
                }
            }


        }
    """).build()

    prgm_grad_weight = cl.Program(ctx, """
        __kernel void conv_backward_weight(__global const float *x_tensor, __global const float *grad_output, __global float *grad_weight,
                                           const int kernel_length, const int num_filters, const int in_channel, const int out_length, 
                                           const int in_length, const int stride, const int batch_size) {

            int filter = (get_global_id(0) / in_channel) % num_filters;
            int channel = get_global_id(0) % in_channel;
            int col = get_global_id(1);

            float sum = 0.0;

            for(int x = 0; x < out_length; x++) {
                for(int b = 0; b < batch_size; b++) {
                    sum += grad_output[b * num_filters * out_length + filter * out_length + x] * \
                        x_tensor[b * in_channel * in_length + channel * in_length + x * stride + col];
                }
            }
            grad_weight[get_global_id(0) * kernel_length + col] = sum;

        }
    """).build()

    grad_x_args = (weight.cl, grad_output.cl, grad_x.cl, np.int32(kernel_length), np.int32(num_filters), 
                   np.int32(in_channel), np.int32(out_length), np.int32(in_length), np.int32(stride), 
                   np.int32(batch_size))
    
    grad_w_args = (x.cl, grad_output.cl, grad_weight.cl, np.int32(kernel_length), np.int32(num_filters), 
                   np.int32(in_channel), np.int32(out_length), np.int32(in_length), np.int32(stride), 
                   np.int32(batch_size))

    prgm_grad_x.conv_backward_x(queue, [batch_size, in_channel], None, *grad_x_args)
    prgm_grad_weight.conv_backward_weight(queue, [num_filters * in_channel, kernel_length], None, *grad_w_args)

    return grad_x, grad_weight
    
    

def conv2d_backward(ctx, queue, grad_output, x, weight, stride):
    batch_size, _, out_height, out_width = grad_output.shape
    num_filters, _, kernel_height, kernel_width = weight.shape
    _, in_channel, im_height, im_width = x.shape

    grad_x = GPUBuffer(ctx, shape=x.shape)
    grad_weight = GPUBuffer(ctx, shape=weight.shape)

    prgm_grad_x = cl.Program(ctx, """
        __kernel void conv_backward_x(__global const float *weight, __global const float *grad_output, __global float *grad_x, const int kernel_height,
                                      const int kernel_width, const int num_filters, const int in_channel, const int out_width, const int out_height,
                                      const int im_width, const int im_height, const int stride, const int batch_size) {

            int batch = get_global_id(0);
            int channel = get_global_id(1);

            for(int y = 0; y < out_height; y++) {
                for(int x = 0; x < out_width; x++) {
                    for(int ky = 0; ky < kernel_height; ky++) {
                        for(int kx = 0; kx < kernel_width; kx++) {
                            
                            float sum = 0.0;

                            for(int f = 0; f < num_filters; f++) {
                                sum += grad_output[batch * num_filters * out_height * out_width + f * out_height * out_width + y * out_width + x] * \
                                    weight[f * in_channel * kernel_height * kernel_width + channel * kernel_height * kernel_width + ky * kernel_width + kx];
                            }

                            grad_x[batch * in_channel * im_height * im_width + channel * im_height * im_width + (y * stride + ky) * im_width + x * stride + kx] += sum;
                        }
                    }
                }
            }
        }
    """).build()

    prgm_grad_weight = cl.Program(ctx, """
        __kernel void conv_backward_weight(__global const float *x_tensor, __global const float *grad_output, __global float *grad_weight,
                                           const int kernel_height, const int kernel_width, const int num_filters, const int in_channel,
                                           const int out_height, const int out_width, const int im_height, const int im_width, const int stride,
                                           const int batch_size) {

            int filter = (get_global_id(0) / in_channel) % num_filters;
            int channel = get_global_id(0) % in_channel;
            int row = get_global_id(1);
            int col = get_global_id(2);

            float sum = 0.0;

            for(int y = 0; y < out_height; y++) {
                for(int x = 0; x < out_width; x++) {
                    for(int b = 0; b < batch_size; b++) {
                        sum += grad_output[b * num_filters * out_height * out_width + filter * out_height * out_width + y * out_width + x] * \
                            x_tensor[b * in_channel * im_width * im_height + channel * im_height * im_width + im_width * (y * stride + row) + x * stride + col];
                    }
                }
            }
            grad_weight[get_global_id(0)*kernel_height*kernel_width + row * kernel_width + col] = sum;
        }
    """).build()

    grad_x_args = (weight.cl, grad_output.cl, grad_x.cl, np.int32(kernel_height), np.int32(kernel_width), np.int32(num_filters), np.int32(in_channel),
                   np.int32(out_width), np.int32(out_height), np.int32(im_width), np.int32(im_height), np.int32(stride), np.int32(batch_size))

    grad_w_args = (x.cl, grad_output.cl, grad_weight.cl, np.int32(kernel_height), np.int32(kernel_width), np.int32(num_filters), np.int32(in_channel),
                   np.int32(out_height), np.int32(out_width), np.int32(im_height), np.int32(im_width), np.int32(stride), np.int32(batch_size))

    prgm_grad_x.conv_backward_x(queue, [batch_size, in_channel], None, *grad_x_args)
    prgm_grad_weight.conv_backward_weight(queue, [num_filters * in_channel, kernel_height, kernel_width], None, *grad_w_args)

    return grad_x, grad_weight
