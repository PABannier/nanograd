import numpy as np
import pyopencl as cl

"""
****** A quick PyOpenCL refresher ******

1) Context creation: the context manages memory allocation. 

2) Queue creation: Every operation executed by OpenCL must pass through 
a queue. PyOpenCL makes sure every command of the program is executed.

3) Arrays are declared a first time at the host level. They must be re-declared
in the C program to be stored in the global memory (i.e. context memory).

4) To store the arrays in the global memory, buffers need to be created in the
context.

5) Program writing: we need to define to be executed by the GPU. __kernel indicates
that the following function is a core function in thhe program. __global means that the
arguments are stored in the global memory. get_global_id(0) allows us to get the index of
the array to process.
"""


