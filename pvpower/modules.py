from torch.autograd import Function
from torch.nn import Module

class MulConstFunc(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None

class AddConstFunc(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        return tensor + constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SumPoolFunc(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.in_size = tensor.size()
        return tensor.sum(dim=(1,2,3))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.view(ctx.in_size[0], 1, 1, 1).expand(ctx.in_size)

class MulConst(Module):

    def __init__(self, const):
        super(MulConst, self).__init__()
        self.const = const

    def forward(self, x):
        return MulConstFunc.apply(x, self.const)

class SumPool(Module):

    def __init__(self):
        super(SumPool, self).__init__()

    def forward(self, x):
        return SumPoolFunc.apply(x)

class AddOne(Module):

    def __init__(self):
        super(AddOne, self).__init__()

    def forward(self, x):
        return AddConstFunc.apply(x, 1)
