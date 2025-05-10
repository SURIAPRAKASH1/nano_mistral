import torch 
import torch.nn as nn

from typing import Type, Tuple
from config import MistralConfig

# simple sin/cos rotary postional embedding 
class MistralRoPE(nn.Module):

  def __init__(self, config: Type[MistralConfig]):
    """
    Rotary Positional Embedding for inserting positional information to tokens
    """
    super(MistralRoPE, self).__init__()

    self.C = config.head_dim
    self.max_block_size = config.max_block_size
    assert self.C % 2 == 0, "RoPE requires Embedding dimension must be even"

    # Precompute RoPE sin/cos matrices

    # compute omega(more like a base angles) (0i= (0, 1, 2, .. n_embd //2))
    half_dim = self.C // 2
    i = torch.arange(half_dim).float()
    omega = 1.0 / (10000 ** (2 * i / self.C))  # (half_dim,)

    # compute postional angles 0(pos)
    pos = torch.arange(self.max_block_size).float()    # (max_block_size,)
    angles = pos[:, None] * omega[None, :]             # (max_block_size, half_dim)

    # compute sin/ cos once
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    self.register_buffer("sin_cached", sin.to(dtype = torch.get_default_dtype()))  # (max_block_size, half_dim)
    self.register_buffer("cos_cached", cos.to(dtype = torch.get_default_dtype))  # (max_block_size, half_dim)

  def forward(self, 
              xq: torch.Tensor, 
              xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, T, C = xq.shape
    assert C % 2 == 0
    half = C // 2

    # select sin/cos for current block_size
    sin = self.sin_cached[:T, :half][None, None, :, :]  # (1, 1, T, half)
    cos = self.cos_cached[:T, :half][None, None, :, :]  # (1, 1, T, half)

    # split xq, xk into odd/even parts
    xq_ = xq.view(*xq.shape[:-1], -1, 2)                # (B, nh, T, half, 2)
    xk_ = xk.view(*xk.shape[:-1], -1, 2)                

    xq_even, xq_odd = xq_[..., 0], xq_[..., 1]          # (B, nh, T, half)
    xk_even, xk_odd = xk_[..., 0], xk_[..., 1]

    # apply rotation (in a multiplication way)
    xq_rot_even = xq_even * cos - xq_odd * sin          # (B, nh, T, half) * (1, 1, T, half) -> (B, nh, T, half)
    xq_rot_odd = xq_even * sin + xq_odd * sin 

    xk_rot_even = xk_even * cos - xk_odd * sin 
    xk_rot_odd = xk_even * sin + xk_odd * sin

    # stack/reshape to get original shape
    xq_rot = torch.stack([xq_rot_even, xq_rot_odd], dim = -1).view(*xq.shape[:-1], -1)  
    xk_rot = torch.stack([xk_rot_even, xk_rot_odd], dim = -1).view(*xk.shape[:-1], -1) 
    return xq_rot, xk_rot


# it's uses complex number's, so it's little complex
class MistralComplexRoPE(nn.Module):
  """
  Rotary Positional Embedding:
    - Rotates the token to propostional to it's position in sequence 
  """

  def __init__(self, config: Type[MistralConfig]):
    super(MistralRoPE, self).__init__()

    self.C = config.head_dim
    self.max_block_size = config.max_block_size  # max seq len
    assert self.C % 2 == 0, "Rope requires embedded token must be even dim"

    freqs = 1.0/(config.omega **( torch.arange(0, self.C, 2)[:self.C//2].float() / self.C ))  # (C//2)
    t = torch.arange(config.max_block_size, device = freqs.device).float()  
    freqs = torch.outer(t, freqs)                       # (max_block_size, C//2)
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # it gives complex64 that can be convert to sin, cos for rope if 

    self.register_buffer('freqs', freqs)

  def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, nh, T, C = xq.shape            # (batch_size, n_head, block_size (seq_len), head_dim)

    half = C // 2
    freqs = self.freqs[None, None, :T, :half]   # (1, 1, :T, :half) 
    
    # make xq, xk as complex num to product with freqs. 
    # to convert complex num we have make xq, xk last dim 2 . cause complex num is real + imag 
    xq_ = torch.view_as_complex( xq.float().reshape(B, nh, T, half, 2) )  # (B, nh, T, half)
    xk_ = torch.view_as_complex( xk.float().reshape(B, nh, T, half, 2) )  # (B, nh, T, half)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)     # (B, nh, T, C)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)