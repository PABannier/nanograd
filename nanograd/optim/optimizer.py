import numpy as np
from nanograd.tensor import Tensor


class Optimizer:
    """Base class for optimizers"""
    def __init__(self, params:list):
        """
            Initializer of the Optimizer class

            Args:
                params (list of Tensors)
        """
        self.params = list(params)
        self.state = []
    
    def step(self):
        """Update rule"""
        raise NotImplementedError("Should be implemented in subclasses of Optimizer")

    def reset(self):
        """Resets all hyperparameters"""
        raise NotImplementedError("Should be implemented in subclasses of Optimizer")

    def zero_grad(self):
        """
            Resets gradients of the parameters 
            to zero. 

            For gradient accumulation, in the training loop, call zero_grad() every
            n (usually 2 or 3) batches
        """
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    """Implementation of Stochastic Gradient Descent (SGD)
    
        Args:
            params (list of Tensors): parameters to be updated
            lr (float): learning rate
            momentum (float): Nesterov momentum
    """
    def __init__(self, params:list, lr:float=1e-3, momentum:float=0) -> None:
        super(SGD, self).__init__(params)
        assert momentum >= 0.0, "Momentum can't be negative"

        self.lr, self.momentum = lr, momentum
        self.momentums = [Tensor.zeros(p.shape, device=self.params[0].device) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            self.momentums[i] = self.momentums[i] * self.momentum + self.lr * p.grad
            p.data = (p - self.momentums[i]).data   


class Adam(Optimizer):
    """Implementation of Adam
    
        Args:
            lr (float): learning rate
            beta1 (float): smoothing factor for first moment of the gradient
            beta2 (float): smoothing factor for second moment of the gradient
            eps (float): avoid division by zero
    """
    def __init__(self, params:list, lr:float=1e-3, beta1:float=0.9, 
                 beta2:float=0.999, eps:float=1e-8) -> None:
        super(Adam, self).__init__(params)
        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"

        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0

        self.exp_avg = [Tensor.zeros(p.shape, device=self.params[0].device) for p in self.params]
        self.exp_avg_sq = [Tensor.zeros(p.shape, device=self.params[0].device) for p in self.params]
    
    def step(self) -> None:
        self.t += 1

        for i, p in enumerate(self.params):
            self.exp_avg[i] = self.beta1 * self.exp_avg[i] + (1. - self.beta1) * p.grad
            self.exp_avg_sq[i] = self.beta2 * self.exp_avg_sq[i] + (1. - self.beta2) * (p.grad ** 2)

            bias_correction1 = self.exp_avg[i] / (1. - (self.beta1 ** self.t))    
            bias_correction2 = self.exp_avg_sq[i] / (1. - (self.beta2 ** self.t))

            p.data = (p - self.lr * bias_correction1 / (bias_correction2.sqrt() + self.eps)).data


class AdamW(Optimizer):
    """Implementation of AdamW: Adam with enhanced weight decay

        Args:
            lr (float): learning rate
            weight_decay (float): L2-regularization parameter
            beta1 (float): smoothing factor for first moment of the gradient
            beta2 (float): smoothing factor for second moment of the gradient
            eps (float): avoid division by zero
    """
    def __init__(self, params:list, lr:float=1e-3, weight_decay:float=1e-2, 
                 beta1:float=0.9, beta2:float=0.999, eps:float=1e-8) -> None:
        super(AdamW, self).__init__(params)
        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"

        self.lr, self.weight_decay, self.eps = lr, weight_decay, eps
        self.beta1, self.beta2 = beta1, beta2
        self.t = 0

        self.exp_avg = [Tensor.zeros(p.shape, device=self.params[0].device) for p in self.params]
        self.exp_avg_sq = [Tensor.zeros(p.shape, device=self.params[0].device) for p in self.params]        

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t

            self.exp_avg[i] = self.beta1 * self.exp_avg[i] + (1. - self.beta1) * p.grad
            self.exp_avg_sq[i] = self.beta2 * self.exp_avg_sq[i] + (1. - self.beta2) * (p.grad ** 2)

            denom = (self.exp_avg_sq[i] / bias_correction2).sqrt() + self.eps
            step_size = self.lr / bias_correction1

            p.data = (p * (1 - self.lr * self.weight_decay) - step_size * (self.exp_avg[i] / denom)).data