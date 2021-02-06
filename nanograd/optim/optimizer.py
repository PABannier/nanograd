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
        self.lr, self.momentum = lr, momentum
        self.momentums = [Tensor.zeros(p.shape) for p in self.params]

        assert self.momentum >= 0.0, "Momentum can't be negative"

    def step(self) -> None:
        """SGD update rule"""
        for param, mom in zip(self.params, self.momentums):
            mom.data = mom.data * self.momentum + self.lr * param.grad.data   # TODO: Find a way to remove .data
            param.data -= mom.data     # Find a way to remove data
            
    def reset(self) -> None:
        self.momentums = [Tensor.zeros(p.shape) for p in self.params]  


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
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

        self._init_parameters()

        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"
    
    def step(self) -> None:
        """Adam update rule"""
        self.t += 1

        for param, m1, m2 in zip(self.params, self.moments1, self.moments2):
            g = param.grad.data
            m1.data = self.beta1 * m1.data + (1 - self.beta1) * g
            m2.data = self.beta2 * m2.data + (1 - self.beta2) * (g ** 2)

            bias_correction1 = m1.data / (1 - (self.beta1 ** self.t))
            bias_correction2 = m2.data / (1 - (self.beta2 ** self.t))

            #param.data -= self.lr * m1_hat / (m2_hat.sqrt() + self.eps)
            param.data -= self.lr * bias_correction1 / (np.sqrt(bias_correction2) + self.eps)

    def reset(self) -> None:
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        self.moments1 = [Tensor.zeros(p.shape) for p in self.params]
        self.moments2 = [Tensor.zeros(p.shape) for p in self.params]
        self.t = 0


class AdamW(Optimizer):
    """
        Implementation of AdamW: Adam with enhanced weight decay

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
        self.lr, self.weight_decay, self.beta1, self.beta2, self.eps = lr, weight_decay, beta1, beta2, eps

        self._init_parameters()

        assert (0 <= beta1) and (beta1 < 1), "Smoothing factor must be in [0,1)"
        assert (0 <= beta2) and (beta2 < 1), "Smoothing factor must be in [0,1)"

    def step(self):
        self.t += 1

        for param, m1, m2 in zip(self.params, self.moments1, self.moments2):
            param.data *= (1 - self.lr * self.weight_decay)
            
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t

            m1.data = m1.data * self.beta1 + param.grad.data * (1 - self.beta1)
            m2.data = m2.data * self.beta2 + (param.grad.data ** 2) * (1 - self.beta2)

            denom = np.sqrt(m2.data / bias_correction2) + self.eps
            step_size = self.lr / bias_correction1

            param.data -= step_size * (m1.data / denom)

    def reset(self) -> None:
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        self.moments1 = [Tensor.zeros(p.shape) for p in self.params]
        self.moments2 = [Tensor.zeros(p.shape) for p in self.params]
        self.t = 0

