from nanograd.tensor import Tensor
from nanograd.device import Device
import inspect

import numpy as np

class Function:
    """
        Superclass for linking nodes to the computational graph.

        Raises:
            NotImplementedError: All subclasses must have a forward and 
            backward function implemented
    """
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *args):
        self.saved_tensors.extend(args)
    
    def apply(self, *args, **kwargs):
        ctx = self(*args)

        params = inspect.signature(self.forward).parameters

        for p in params.values():
            if p.default is not p.empty:
                setattr(ctx, p.name, p.default)
        
        for k, v in kwargs.items():
            setattr(ctx, k, v)

        
        out = self.forward(ctx, *[t.data for t in args], **kwargs)
        ret = Tensor(out, device=ctx.device, requires_grad=any([t.requires_grad for t in args]))
        
        if ret.requires_grad:
            ret.ctx = ctx
        
        return ret
