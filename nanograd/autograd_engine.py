import tensor

def backward(grad_fn, grad_of_outputs):
    r"""
        Recursive DFS that traverses comp graph, handing back gradients as it goes.

        Args:
            grad_fn (BackwardFunction or AccumulateGrad): Current node type from
                                                        parent's `.next_functions`
            grad_of_output (Tensor): Gradient of the final node w.r.t. current output

        Returns:
            No return statement needed.
    """

    if grad_fn:
        gradients = grad_fn.apply(grad_of_outputs)
        functions = grad_fn.next_functions
        
        for i in range(len(functions)):
            if functions[i]:
                backward(functions[i], gradients[i])

class Function:
    r"""
        Superclass for linking nodes to the computational graph.
        Operations in `functional.py` should inherit from this
    """
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("All subclasses must implement backward")

    @classmethod
    def apply(cls, *args):
        r"""
            Runs forward of subclass and links node to the comp graph.

            Args:
                cls (subclass of Function): Current function, such as Add, Sub, etc.
                args (tuple): arguments for the subclass's `.forward()`.

            Returns:
                Tensor: Output tensor from operation that stores the current node.
        """
        # Creates BackwardFunction obj representing the current node
        backward_function = BackwardFunction(cls)

        # Run subclass's forward with context manager and operation input args
        output_tensor = cls.forward(backward_function.ctx, *args)

        # 1) For each parent tensor in args, add their node to `backward_function.next_functions`
        for arg in args:
            if isinstance(arg, tensor.Tensor):
                if not arg.grad_fn:
                    if arg.requires_grad and arg.is_leaf:
                        arg.grad_fn = AccumulateGrad(arg)
                    elif arg.requires_grad and not arg.is_leaf:
                        arg.grad_fn = BackwardFunction(cls)
                    elif not arg.requires_grad and arg.is_leaf:
                        arg.grad_fn = None
                    else:
                        raise Exception("Gradient-disabled nodes are necessarily leaves")
                
                backward_function.next_functions.append(arg.grad_fn)
            else:
                backward_function.next_functions.append(None)

        # 2) Store current node in output tensor (see `tensor.py` for ideas)
        output_tensor.grad_fn = backward_function

        return output_tensor


class AccumulateGrad:
    r"""
        Represents node where gradient must be accumulated.

        Args:
            tensor (Tensor): The tensor where the gradients are accumulated in `.grad`
    """
    def __init__(self, tensor):
        self.variable = tensor
        self.next_functions = [] 
        self.function_name = "AccumulateGrad"

    def apply(self, arg):
        r"""
            Accumulates gradient provided.

            Args:
                arg (Tensor): Gradient to accumulate
        """
        # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = tensor.Tensor(arg.data)
        else:
            self.variable.grad.data += arg.data

        # Some tests to make sure valid grads were stored.
        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        assert shape == grad_shape, (shape, grad_shape)

class ContextManager:
    r"""
        Used to pass variables between a function's `.forward()` and `.backward()`.
        (Argument "ctx" in these functions)

        To store a tensor:
        >>> ctx.save_for_backward(<tensors>, <to>, <store>)

        To store other variables (like integers):
        >>> ctx.<some_name> = <some_variable>
    """
    def __init__(self):
        self.saved_tensors = [] 

    def save_for_backward(self, *args):
        r"""
            Saves TENSORS only

            Args:
                args (Tensor(s)): Tensors to store
        """
        for arg in args:
            # Raises error if arg is not tensor (i warned you)
            if type(arg).__name__ != "Tensor":
                raise Exception("Got type {} of object {}. \nOnly Tensors should be saved in save_for_backward. For saving constants, just save directly as a new attribute.".format(type(arg), arg))

            self.saved_tensors.append(arg.copy())


class BackwardFunction:
    r"""
        Representing an intermediate node where gradient must be passed.
        Stored on output tensor of operation during `Function.apply()`

        Args:
            cls (subclass of Function): Operation being run. Don't worry about this;
                                        already handled in `Function.apply()`
    """
    def __init__(self, cls):
        self.ctx = ContextManager()
        self._forward_cls = cls

        # Nodes of parents, populated in `Function.apply`
        self.next_functions = []

        # The name of the operation as a string (for convenience)
        self.function_name = cls.__name__

    def apply(self, *args):
        r"""
            Generates gradient by running the operation's `.backward()`.

            Args:
                args: Args for the operation's `.backward()`

            Returns:
                Tensor: gradient of parent's output w.r.t. current output
        """
        return self._forward_cls.backward(self.ctx, *args)
