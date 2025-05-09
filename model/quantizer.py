import torch
import torch.nn as nn


"""
so if we don't want quantized llm then it's better to use quant = False in MistralConfig.

Used to dequantize the weights that already in compressed format
Here so much stuff going on it's better to not use quant . only helpfull if we have low end spec 
eg:
    qweight - 8 or 4 bit int that packed into int 32 but actuall weight size usually float 16 or 32
"""

class DeQuantizer(nn.Module):
 
  def __init__(self,
               bits,
               qzeros,
               qweight,
               pack_factor,
               maxq,
               scales,
               g_idx,
               dequant_dtype,
               pack_dtype_bits
               ):
    super().__init__()

    self.bits = bits          # bit size that used to compressed our weight
    self.qzeros = qzeros
    self.qweight = qweight
    self.scales = scales
    self.pack_factor =pack_factor
    self.dequant_dtype = dequant_dtype
    self.maxq = maxq
    self.g_idx = g_idx
    self.pack_dtype_bits = pack_dtype_bits

    if self.bits in [2, 4, 8]:
        wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype= torch.int32).unsqueeze(0).to(
              device=self.g_idx.device)

    self.wf_unsqueeze_zero = wf.unsqueeze(0).to(device=self.g_idx.device)
    self.wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(device=self.g_idx.device)

  def dequantize_weight(self, num_itr: int = 1):
    
    if self.bits in [2, 4, 8]:
        zeros =torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros =torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight =torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if num_itr == 1:
        weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
    else:
        num_dim = self.g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = self.scales[:, i * num_dim: (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim: (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim: (i + 1) * num_dim]
            g_idx_i = self.g_idx[i * num_dim: (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights =torch.cat(weights, dim=1)

    return weights

class TorchQuantLinear(nn.Module):

  def __init__(self, 
               n_embd,
               out_features, #(n_head * head_id)m
               bias,
               bits = 4, 
               pack_factor = 8,
               maxq = 15,
               dequant_dtype = torch.int8,
               pack_dtype_bits = 32,
               group_size = 128,
               ):
    super().__init__()

    self.out_features = out_features
    # just a palce holder for pretrained quantized weights
    self.qweight = nn.Parameter(torch.empty(n_embd // 8, out_features, dtype = torch.int32), requires_grad= False) 
    self.scales = nn.Parameter(torch.ones(n_embd // group_size, out_features), requires_grad= False)
    self.qzeros = nn.Parameter(torch.empty(n_embd // group_size, out_features // 8, dtype= torch.int32), requires_grad= False) 
    self.g_idx = nn.Parameter(torch.empty(n_embd, dtype= torch.int32), requires_grad= False) 
    self.bias = bias # no needed

    self.dequantizer = None
    self.bits = bits
    self.pack_factor = pack_factor
    self.maxq = maxq 
    self.dequant_dtype = dequant_dtype 
    self.pack_dtype_bits = pack_dtype_bits
    self.group_size = group_size
   
    self.bias = None
    self.adapter = None
    
  def finalize_dequantizer(self):
    self.dequantizer = DeQuantizer(
              self.bits,
              self.qzeros,
              self.qweight,
              self.pack_factor,
              self.maxq,
              self.scales,
              self.g_idx,
              self.dequant_dtype,
              self.pack_dtype_bits
        
    )

  def forward(self, x: torch.Tensor):
    if self.dequantizer is None:
      self.finalize_dequantizer()

    out_shape = x.shape[:-1] + (self.out_features,)
    x = x.reshape(-1, x.shape[-1])
    out = self._forward(x, out_shape)
    return out

  def _forward(self, x, out_shape):
    num_itr = self.g_idx.shape[0] // x.shape[-1]
    # make sure dequant dtype matches input x
    weights = self.dequantizer.dequantize_weight(num_itr=num_itr).to(x.dtype)

    out = torch.matmul(x, weights).reshape(out_shape)

    if self.bias is not None:
        out.add_(self.bias)

    if self.adapter:
        out = self.adapter.apply(x=x, out=out)

    return out
