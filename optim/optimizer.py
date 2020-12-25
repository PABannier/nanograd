import numpy as np
from tensor import Tensor

class Optimizer:
    r"""
        Base class for optimizers
    """
    def __init__(self, params):
        r"""
            Initializer of the Optimizer class

            Args:
                params (list of Tensors)
        """
        self.params = list(params)
        self.state = []
    
    def step(self):
        r""" Update rule"""
        raise NotImplementedError("Should be implemented in subclasses of Optimizer")

    def zero_grad(self):
        r"""
            Resets gradients of the parameters 
            to zero. 

            For gradient accumulation, in the training loop, call zero_grad() every
            n (usually 2 or 3) batches
        """
        for param in self.params:
            param.grad = None

class SGD(Optimizer):
    r"""
        Implementation of Stochastic Gradient Descent (SGD)
    """
    def __init__(self, params, lr=1e-2, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.momentums = [Tensor.zeros(p.shape) for p in self.params]

        assert self.momentum >= 0.0, "Momentum can't be negative"

    def step(self):
        r"""SGD update rule"""
        for param, mom in zip(self.params, self.momentums):
            mom.data = mom.data  * self.momentum + self.lr * param.grad.data
            param.data -= mom.data        
