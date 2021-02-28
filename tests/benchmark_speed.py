from nanograd.tensor import Tensor
from tests.helpers import create_identical_torch_tensor

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

import timeit
import unittest

BENCHMARKS = {}

def create_tensor_with_shapes(shapes):
    tensors = []
    for shape in shapes:
        t = Tensor.normal(30, 1, shape)
        tensors.append(t)
    return tuple(tensors)

def benchmark_speed_ops(shapes, op_nanograd, op_pytorch=None, name='None'):
    def profile_op(tensors, op):
        return timeit.timeit(lambda: op(*tensors), number=10)

    tensors = create_tensor_with_shapes(shapes)
    
    ng_cpu_time = profile_op(tensors, op_nanograd)

    op_pytorch = op_nanograd if op_pytorch is None else op_pytorch
    pt_tensors = [create_identical_torch_tensor(t) for t in tensors]
    pt_cpu_time = profile_op(pt_tensors, op_pytorch)

    #tensors = [t.gpu() for t in tensors]
    #ng_gpu_time = profile_op(tensors, op_nanograd)

    BENCHMARKS[name] = (ng_cpu_time, pt_cpu_time)

def plot_results():
    labels = list(BENCHMARKS.keys())
    cpu_speed = [x[0]*1000 for x in BENCHMARKS.values()]
    gpu_speed = [x[1]*1000 for x in BENCHMARKS.values()]

    x = np.arange(len(labels)) 
    width = 0.35
     
    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width/2, cpu_speed, width, label='CPU')
    rects2 = ax.bar(x + width/2, gpu_speed, width, label='PyTorch')

    ax.set_ylabel('Time (in ms)')
    ax.set_title('Speed benchmark - Nanograd ops')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.legend()
    fig.tight_layout()
    plt.show()
    

class BenchmarkSpeedOps(unittest.TestCase):
    def test_basic_op_speed(self):
        BENCHMARKS = dict()
        benchmark_speed_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x+y, name='add')
        benchmark_speed_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x-y, name='sub')
        benchmark_speed_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x*y, name='mul')
        benchmark_speed_ops([(10, 15, 3), (10, 15, 3)], lambda x,y: x/y, name='div')
        benchmark_speed_ops([(30, 40), (40, 20)],       lambda x,y: x@y, name='matmul')
        plot_results()

    def test_broadcast_speed(self):
        BENCHMARKS = dict()
        benchmark_speed_ops([(10, 10), (10, 1)],        lambda x,y: x+y, name='add_broadcast')
        benchmark_speed_ops([(10, 10), (10, 1)],        lambda x,y: x-y, name='sub_broadcast')
        benchmark_speed_ops([(10, 10), (10, 1)],        lambda x,y: x*y, name='mul_broadcast')
        benchmark_speed_ops([(10, 10), (10, 1)],        lambda x,y: x/y, name='div_broadcast')
        plot_results()

    def test_unary_op_speed(self):
        BENCHMARKS = dict()
        benchmark_speed_ops([(10, 15, 3)],              lambda x: -x, name='neg')
        benchmark_speed_ops([(20, 20)],                 lambda x: x ** 3, name='pow')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.log(), name='log')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.exp(), name='exp')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.sqrt(), name='sqrt')
        benchmark_speed_ops([(256, 2)],                 lambda x: x.reshape(shape=[8, 8, 8]), name='reshape_2')
        benchmark_speed_ops([(20, 10)],                 lambda x: x.T, lambda x: x.T, name='transpose')
        benchmark_speed_ops([(30, 40, 20)],             lambda x: x.sum(axis=(1, 2)), name='sum_3')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.mean(axis=0), name='mean')
        benchmark_speed_ops([(30, 40, 20, 10)],         lambda x: x.min(axis=3).min(axis=0), lambda x: x.min(axis=3)[0].min(axis=0)[0], name='min')
        benchmark_speed_ops([(30, 40, 20, 10)],         lambda x: x[10:20, 4:5, :12, 0:-1], name='getitem_2')
        benchmark_speed_ops([(30, 30, 1)],             lambda x: x.squeeze(axis=2), name='squeeze_1')
        benchmark_speed_ops([(30, )],                   lambda x: x.unsqueeze(axis=1), name='unsqueeze')
        plot_results()
    
    def test_activation_speed(self):
        BENCHMARKS = dict()
        benchmark_speed_ops([(20, 20)],                 lambda x: x.relu(), name='relu')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.tanh(), name='tanh')
        benchmark_speed_ops([(20, 20)],                 lambda x: x.sigmoid(), name='sigmoid')
        benchmark_speed_ops([(128, 10)],                lambda x: x.log_softmax(), lambda x: F.log_softmax(x, dim=1), name='log_softmax')
        plot_results()
    
    def test_conv1d_speed(self):
        BENCHMARKS = dict()
        for pad_size in range(5):
            benchmark_speed_ops([(20, 3, 100)], lambda x: x.pad1d((pad_size, pad_size)), lambda x: F.pad(x, (pad_size, pad_size)), name=f'pad1d_{pad_size}')
        for pool_size in range(1, 4):
            benchmark_speed_ops([(20, 3, 100)], lambda x: x.max_pool1d(pool_size), lambda x: F.max_pool1d(x, pool_size), name=f'mpool1d_{pool_size}')
        for pool_size in range(1, 4):
            benchmark_speed_ops([(20, 3, 100)], lambda x: x.avg_pool1d(pool_size), lambda x: F.avg_pool1d(x, pool_size), name=f'avgpool1d_{pool_size}')
        for kernel_size in range(2, 6):
            for stride in range(1, 5):
                benchmark_speed_ops([(20, 3, 100), (10, 3, kernel_size)], lambda x, w: x.conv1d(w, stride), 
                                    lambda x, w: F.conv1d(x, w, stride=stride), name=f'conv1d_{kernel_size}_{stride}')
        plot_results()
    
    def test_conv2d_speed(self):
        BENCHMARKS = dict()
        benchmark_speed_ops([(20, 3, 30, 30)], lambda x: x.pad2d((1, 2, 3, 4)), lambda x: F.pad(x, (1, 2, 3, 4)), name='pad2d')
        for pool_size in range(1, 5):
            benchmark_speed_ops([(20, 3, 30, 30)], lambda x: x.max_pool2d((pool_size, pool_size)), 
                                lambda x: F.max_pool2d(x, kernel_size=(pool_size, pool_size)), name=f'mpool2d_{pool_size}')
        for pool_size in range(1, 5):
            benchmark_speed_ops([(20, 3, 30, 30)], lambda x: x.avg_pool2d((pool_size, pool_size)), 
                                lambda x: F.avg_pool2d(x, kernel_size=(pool_size, pool_size)), name=f'avgpool2d_{pool_size}')
        for kernel_size in range(2, 6):
            for stride in range(1, 5):
                benchmark_speed_ops([(20, 3, 30, 30), (10, 3, kernel_size, kernel_size)], lambda x, w: x.conv2d(w, stride), 
                                    lambda x, w: F.conv2d(x, w, stride=stride), name=f'conv2d_{kernel_size}_{stride}')
        plot_results()
        