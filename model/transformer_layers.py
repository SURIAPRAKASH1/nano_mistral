import torch 
import torch.nn as nn

import importlib
from typing import Optional

from quantizer import TorchQuanLinear
from lora import LoRaLinear
from config import MistralConfig


def multi_linear(
    config: Optional[MistralConfig]
    ):
  """
  There are three layer:
    1. Linear -> as usuall
    2. QuantLinear -> it will be used if we choose to use pretrained quantized weights
    3. LoRaLinear -> lower rank adaptation layer that used to freeze pretrained weights and
                     learn other weights then added to pretrained weights. very efficient when fine-tuning
  """
  if config.lora:
    return LoRaLinear
  elif config.quant:
    return TorchQuantLinear
  else:
    return nn.Linear


class MistralAttention(nn.Module):
  """
  Mistral leverage GQA and SWQ:
    1. GQA-->(Group Query Attention). where group of query's (in defferent heads) communicate with same key-value
    2. SWA-->(Sliding Window Attention). tokens only communicate with other tokens that are in certion window

  """

  def __init__(self, config: Optional[MistralConfig]):
    super(MistralAttention, self).__init__()
    
    self.n_embd = config.n_embd
    self.head_dim = config.head_dim
    self.n_head = config.n_head
    self.n_kv_head = config.n_kv_head
    self.repeats = self.n_head // self.n_kv_head
    self.dropout = config.dropout
    self.block_size = config.block_size
    self.window_size = config.window_size 

    # so here we can pack matrix togather but in future we(I) may use pretrained mistral weights 😁
    MultiLinear = multi_linear(config)
    self.q_proj = MultiLinear(self.n_embd, self.n_head * self.head_dim, bias = config.bias)
    self.k_proj = MultiLinear(self.n_embd, self.n_kv_head * self.head_dim, bias = config.bias)
    self.v_proj = MultiLinear(self.n_embd, self.n_kv_head * self.head_dim, bias = config.bias)
    self.o_proj = MultiLinear(self.head_dim * self.n_head, self.n_embd, bias = config.bias)


    # regularizaiton
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

    self.device = self.q_proj.weight.device if isinstance(self.q_proj, nn.Linear) else self.q_proj.qweight.device

    # effcient implemention of GQA using xformers. xformers requires cuda
    self.xformer = True if importlib.util.find_spec('xformers') and self.device == 'cuda' else False

    if not self.xformer:
      self.register_buffer('attn_bias', 
                            swa_mask(self.block_size, self.window_size, self.device)
                            .view(1, 1, self.block_size, self.block_size))

  def forward(self, x: torch.Tensor):
    B, T, C = x.shape               # (B, T, C) --> (batch_size, block_size, n_embdding_dim)

    # project and do some fused shape modification operations
    q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)           # (B, nh, T, hd)
    k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)        # (B, n_kv_h, T, hd)
    v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)        # (B, n_kv_h, T, hd)

    # GQA for group of q from different head attention to same k,v from same head
    # repeat k,v heads to match q head. means just make copy of kv
    k = torch.repeat_interleave(k, self.repeats, dim = 1)                 # (B, n_kv_h * repeats, T, hd)
    v = torch.repeat_interleave(v, self.repeats, dim = 1)                 # (B, n_kv_h * repeats, T, hd)

    if self.xformer:
      # efficient kernal operation. here we changing shape cause xformer need to have (B, T, nh, hd)
      attn = fmha.memory_efficient_attention(
          q.transpose(1, 2),
          k.transpose(1, 2),
          v.transpose(1, 2),
          p = self.dropout,
          attn_bias= fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(_window_size= self.window_size)
      )

    else:
      # haha just a manuall implementation
      attn_scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)        # (B, nh, T, hd) @ (B, nh, hd, T) --> (B, nh, T, T)
      attn_scores = attn_scores.masked_fill_(self.attn_bias[:, :, :T, :T] == 0 , float('-inf'))
      attn_weight = torch.softmax(attn_scores, dim = -1) 
      attn_weight = self.attn_dropout(attn_weight)
      attn = attn_weight @ v                                            # (B, nh, T, T)  @ (B, nh, T, hd) --> (B, nh, T, hd)
      attn = attn.transpose(1, 2).contiguous()                          # (B, T, hd, hd)

    # collapse and reshape attn to get same shape as input and do finall projection
    output = attn.view(B, T, C)      # (B, T, nh * hd)
    output = self.o_proj(self.resid_dropout(output))

    return output


class MistralMLP(nn.Module):
  """Fully connected dense layer"""

  def __init__(self, config):
    super(MistralMLP, self).__init__() 

    MultiLinear = multi_linear(config) 
    # funny thing about this layer is we usually 4 * n_embd for intermediate dim of mlp but mistral did 3.5 times 😀 
    self.up_proj = MultiLinear(config.n_embd, config.mlp_hidden, config.bias) 
    self.down_proj = MultiLinear(config.mlp_hiddennt, config.n_embd, config.bias) 
    
    # here we making mistake 
    self.act = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.act(self.up_proj(x)))

