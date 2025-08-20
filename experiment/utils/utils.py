import torch
import numpy as np
import random
from torch import nn
from torch.autograd import Function

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_and_write(message: str, file=None):
    """打印并写入日志文件"""
    print(message)
    if file:
        with open(file, 'a') as f:
            f.write(message + '\n')

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from DANN paper by Ganin et al.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass is an identity function.
        :param ctx: context object to store intermediate values
        :param x: input tensor
        :param lambda_: hyperparameter for gradient scaling
        :return: original input tensor
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass applies the gradient reversal.
        :param ctx: context object to retrieve stored values
        :param grad_output: gradient of the loss with respect to the output
        :return: gradient of the loss with respect to the input
        """
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None
    
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)