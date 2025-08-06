import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 实现
    为Transformer模型提供相对位置编码
    """
    
    def __init__(self, dim):
        super().__init__()
        # 计算逆频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, max_seq_len, *, device):
        """
        生成旋转位置嵌入
        Args:
            max_seq_len: 最大序列长度
            device: 设备
        Returns:
            旋转位置嵌入张量
        """
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i,j->ij", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    """
    旋转张量的一半维度
    Args:
        x: 输入张量
    Returns:
        旋转后的张量
    """
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    """
    应用旋转位置嵌入
    Args:
        pos: 位置嵌入
        t: 输入张量
    Returns:
        应用了旋转位置嵌入的张量
    """
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class RotaryPositionalEmbedding(nn.Module):
    """
    完整的RoPE模块，包含位置嵌入生成和应用
    """
    
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rotary_emb = RotaryEmbedding(dim)
        
    def forward(self, x, seq_len=None):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, num_heads, head_dim]
            seq_len: 序列长度，如果为None则使用x的序列长度
        Returns:
            应用了RoPE的张量
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # 生成位置嵌入
        pos_emb = self.rotary_emb(seq_len, device=x.device)
        
        # 应用旋转位置嵌入
        return apply_rotary_pos_emb(pos_emb, x) 