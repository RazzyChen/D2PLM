import torch
import torch.nn as nn

def swiglu(x):
    """
    SwiGLU激活函数实现
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """
    SwiGLU激活函数模块
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return swiglu(x)
