from fastai.torch_core import *

__all__ = ['MaskedFlattenedLoss', 'MaskedFlattenedCrossEntropy']


class MaskedFlattenedLoss:
    "Same as `func`, but 1) flattens input and target and 2) ignores all values where target < 0."
    def __init__(self, func, *args, axis:int=-1, floatify:bool=False, is_2d:bool=True, **kwargs):
        self.func,self.axis,self.floatify,self.is_2d = func(*args,**kwargs),axis,floatify,is_2d
        functools.update_wrapper(self, self.func)

    def __repr__(self): return f"MaskedFlattenedLoss of {self.func}"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def __call__(self, input:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        input = input.transpose(self.axis,-1).contiguous()
        target = target.transpose(self.axis,-1).contiguous()
        if self.floatify: target = target.float()
        input = input.view(-1,input.shape[-1]) if self.is_2d else input.view(-1)
        target = target.view(-1)
        mask = target >= 0
        target = target[mask]
        input = input[mask]
        return self.func.__call__(input, target, **kwargs)


MaskedFlattenedCrossEntropy = partial(MaskedFlattenedLoss, nn.CrossEntropyLoss)


