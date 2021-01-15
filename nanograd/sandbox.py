import numpy as np
import pyopencl as cl

def get_gpu_context_and_queue():
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
        devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)
    return cl_ctx, cl_queue

def get_output_shape(in_shape:tuple, axis:int=None, keepdims:bool=False) -> np.ndarray:
    in_shape = np.array(in_shape)
    if keepdims:
      in_shape[axis] = 1
      return in_shape
    return np.delete(in_shape, axis) if axis is not None else (1,)

def reduce_op(ctx, code, code2, inp, axis=None, start="0.0"):
    out_shape = get_output_shape(inp.shape, axis=axis)
    out_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                        4*np.prod(out_shape), hostbuf=None)
    
    reduce = clbuild(ctx, "reduce", """
        __kernel void reduce(__global const float *a_g, int sz, __global float *res_g, int prod, 
                             int n_dims, __global const int *shape_x, __global const int *shape_ret) {
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
        }
      """)
    reduce(ctx.cl_queue, [np.prod(osize)], None, inp,
      np.int32(np.prod(inp.shape)//np.prod(out_shape)), out_buf,
      np.int32(np.prod(out_shape)), np.int32(len(out_shape)),
      buffer_np(ctx, np.array(inp.shape, dtype=np.int32)),
      buffer_np(ctx, np.array(osize, dtype=np.int32)))
    return ret



"""
ctx, queue = get_gpu_context_and_queue()
n_threads = ctx.get_info(cl.context_info.DEVICES)[0].max_work_group_size

in_array = np.random.normal(0, 1, size=(10000, 1)).astype(np.float32)
out_array = np.empty(n_threads).astype(np.float32)
output = np.empty(1).astype(np.float32)

mf = cl.mem_flags
in_arr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=in_array)
out_arr_buf = cl.Buffer(ctx, mf.READ_WRITE, size=out_array.nbytes)
out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)
loc_buf = cl.LocalMemory(4*n_threads)

prg = cl.Program(ctx, 
    __kernel void reduce(__global float *a, __global float *r, __local float *b) {
        uint gid = get_global_id(0);
        uint wid = get_group_id(0);
        uint lid = get_local_id(0);
        uint gs = get_local_size(0);

        b[lid] = a[gid];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint s = gs/2; s > 0; s >>= 1) {
          if(lid < s) {
            b[lid] += b[lid+s];
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(lid == 0) r[wid] = b[lid];
    }
    ).build()

evt = prg.reduce(queue, (10000,), (n_threads,), in_arr_buf, out_arr_buf, loc_buf)
evt.wait()
evt = prg.reduce(queue, (n_threads,), (n_threads,), in_arr_buf, out_arr_buf, loc_buf)
evt.wait()
cl.enqueue_copy(queue, output, out_buf)

print(np.allclose(output, np.sum(in_array)))
print(np.sum(in_array))
print(output)
"""