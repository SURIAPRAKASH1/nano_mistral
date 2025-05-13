import torch 
import torch.nn as nn
import torch.nn.functional as F 

from typing import Type, Optional 

from model.config import MistralConfig
from model.transformer_layers import MistralBlock, MistralRMSNorm


class MistralTransformer(nn.Module):

  def __init__(self, config: Type[MistralConfig]):
    super(MistralTransformer, self).__init__()

    self.vocab_size = config.vocab_size
    self.n_embd = config.n_embd
    self.max_block_size = config.max_block_size 
    self.block_size  = config.block_size 

    # token embedding
    self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, dtype = config.dtype)
    self.drop = nn.Dropout(config.dropout) 

    # transformer layers
    self.layers = nn.ModuleList([
        MistralBlock(config) for _ in range(config.n_layer)
    ])

    # weight initialization
    self.apply(self._init_weight) 

    # final normalization before getting logits
    self.norm = MistralRMSNorm(config.n_embd, dtype= config.dtype , eps = 1e-6)

    # final logits projection
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False, dtype = config.dtype)

    # get parameters count
    print('total parameters %.2fM' % (self._get_parameters_count()/1e+6))

  def device(self):
    return next(self.parameters()).device

  def _init_weight(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean = 0.0, std = 0.2)
      if module.bias is not None: 
        nn.init.zeros(module.bias) 
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean = 0.0, std = 0.2)

  def _get_parameters_count(self):
    # here we don't use weight dying like in gpt's so have to count lm_head as well
    tp = 0
    for p in self.parameters():
      tp += p.nelement()
    return tp

  def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
    B, T = input_ids.shape 
    assert T <= self.block_size, f"can't forward seq len is {T}, But max block size is {self.max_block_size}"

    h = self.embed_tokens(input_ids)   # token embeddings (B, T, C)
    h = self.drop(h)                   # dropout before feeding to transformer

    for layer in self.layers:
      h = layer(h)
    h = self.norm(h)           # normalization before logits

    # optionally if i we have targets then we can compute loss
    if target_ids is not None:
      logits = self.lm_head(h)
      loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))
    else:
      # inferece we have predict token by taking last token only even though model learned full seq.
      # like predicting the next word in story after read
      logits = self.lm_head(h[:, [-1], :])
      loss = None

    return logits, loss

  def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    only consider tokens that have high probability and their cumulative prob's is higher than p
    eg: probs = [0.01, 0.02, 0.07, 0.1, 0.5, 0.3]. only sample from probs = [0.5, 0.3, 0.07] if p > 0.9
    """
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim = -1, descending= True)   # sort the probs from high to low 
    probs_sum = torch.cumsum(probs_sort, dim = -1)                          # cumulate the probs
    mask = (probs_sum - probs_sort)  > p                                    # mask for removing low prob's tokens
    probs_sort[mask] = 0.0 
    probs_sort.div_(probs_sort.sum(-1, keepdim= True))                      # we have normalized those top p 
    next_token = torch.multinomial(probs_sort, num_samples = 1)
    return torch.gather(probs_idx, -1, next_token) 


  def generate(self,
               ids: torch.Tensor,
               max_tokens:int = 50,
               temprature:Optional[float] = None,
               top_k:Optional[int] = None, 
               top_p:Optional[float] = None) -> torch.Tensor:

    """
    Parameters:
      ids -> Tensor with shape (B, T), dtype int . here B-bach_size, T-seq_len
      max_tokens -> How many new tokes we want to get
      temprature -> optional . if high means we get more diverce predict. if low means it greedy
      top_k -> optionaly to get probabilities for tokens from top_k of logits
      top_p -> optionaly to get tokens only from high probabilities
      ** if none of the temprature, top_k, top_p then it's greedy approach. only get tokens that is most likly. not recommended **
      ** temprature = 0.7, top_k = 50, top_p = 0.8 is good way to start **
    Output:
      ids -> Tensor with shape (B, T + max_tokens)
    """
    with torch.no_grad():
      for i in range(max_tokens):
        # we have to crop the seqlen if it's exit model block size
        ids_crop = ids if ids.size(1) <= self.block_size else ids[:, -self.block_size:]   # (B, T)
        # feed to model to get logits over next tokens
        logits, _ = self(ids_crop)
        logits = logits[:, -1, :]               # (1, vocab_size) 
        
        if temprature is not None:
          logits = logits / temprature            # controling the randomess. high temprature means high diverce

          if top_k is not None:
            assert top_k < self.vocab_size 
            topk, _ = torch.topk(logits, top_k, dim = -1)
            logits[logits < topk[:, [-1]]] = float('-inf')    # logits < top_k will be -inf. when do softmax it will be 0.0.

          # get distripution over next tokens
          probs = torch.softmax(logits , dim = -1, dtype = torch.float32) 
          next_tok = torch.multinomial(probs, 1)

          if top_p is not None:
              next_tok =  self._sample_top_p(probs, top_p)
        else:
          next_tok = torch.argmax(logits, dim = -1, keepdim = True) 

        # auto aggression manner
        ids = torch.cat([ids, next_tok], dim = -1)   # every iter (B, T+1)

    return ids
