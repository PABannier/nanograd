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
        r"""Update rule"""
        raise NotImplementedError("Should be implemented in subclasses of Optimizer")

    def reset(self):
        r"""Resets all hyperparameters"""
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
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params)
        self.lr, self.momentum = lr, momentum
        self.momentums = [Tensor.zeros(p.shape) for p in self.params]

        assert self.momentum >= 0.0, "Momentum can't be negative"

    def step(self):
        r"""SGD update rule"""
        for param, mom in zip(self.params, self.momentums):
            mom.data = mom.data  * self.momentum + self.lr * param.grad.data
            param.data -= mom.data

    def reset(self):
        self.momentums = [Tensor.zeros(p.shape) for p in self.params]  


class Adam(Optimizer):
    r"""
        Implementation of Adam
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        r"""
            Args:
                lr: learning rate
                beta1: smoothing factor for first moment of the gradient
                beta2: smoothing factor for second moment of the gradient
                eps: avoid division by zero
        """
        super().__init__(params)
        self.lr, self.beta1, self.bet2, self.eps = lr, beta1, beta2, eps

        self._init_parameters()

        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"
    
    def step(self):
        r"""Adam update rule"""
        self.t += 1

        for param, m1, m2 in zip(self.params, self.moments1, self.moments2):
            g = param.grad.data
            m1.data = self.beta1 * m1.data + (1 - self.beta1) * g
            m2.data = self.beta2 * m2.data + (1 - self.beta2) * (g ** 2)

            m1_hat = m1.data / (1 - (self.beta1 ** self.t))
            m2_hat = m2.data / (1 - (self.beta2 ** self.t))

            param.data -= self.lr * m1_hat / (np.sqrt(m2_hat) + self.eps)

    def reset(self):
        self._init_parameters()
    
    def _init_parameters(self):
        self.moments1 = [Tensor.zeros(p.shape) for p in self.params]
        self.moments2 = [Tensor.zeros(p.shape) for p in self.params]
        self.t = 0


class AdamW(Optimizer):
    r"""
        Implementation of AdamW

        Adam with enhanced weight decay
    """
    def __init__(self, params, lr=1e-3, reg=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        r"""
            Args:
                lr: learning rate
                reg: L2-regularization parameter
                beta1: smoothing factor for first moment of the gradient
                beta2: smoothing factor for second moment of the gradient
                eps: avoid division by zero
        """
        super().__init__(params)
        self.lr, self.reg, self.beta1, self.beta2, self.eps = lr, reg, beta1, beta2, eps

        self._init_parameters()

        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"
    
    def step(self):
        r"""Adam update rule"""
        self.t += 1

        for param, m1, m2 in zip(self.params, self.moments1, self.moments2):
            g = param.grad.data + self.reg * param.data
            m1.data = self.beta1 * m1.data + (1 - self.beta1) * g
            m2.data = self.beta2 * m2.data + (1 - self.beta2) * (g ** 2)

            m1_hat = m1.data / (1 - (self.beta1 ** self.t))
            m2_hat = m2.data / (1 - (self.beta2 ** self.t))

            param.data -= ((self.lr * m1_hat / (np.sqrt(m2_hat) + self.eps)) + self.reg * param.data)

    def reset(self):
        self._init_parameters()
    
    def _init_parameters(self):
        self.moments1 = [Tensor.zeros(p.shape) for p in self.params]
        self.moments2 = [Tensor.zeros(p.shape) for p in self.params]
        self.t = 0

