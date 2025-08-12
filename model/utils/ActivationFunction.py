import torch

class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gates) * x