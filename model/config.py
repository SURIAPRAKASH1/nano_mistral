from dataclasses import dataclass


@dataclass
class MistralConfig:

  n_embd: int = 128 #4096                        # token embedding (just a vector representation haa) dim
  vocab_size:int = 32000                   # total unique tokens
  block_size: int = 24 #8196                     # just a seq_len.
  batch_size: int = 32                     # how may block (seq) are are packed togather
  n_head: int = 8 #32                          # How many heads?
  n_kv_head: int = 2 #8                       # if we use GQA then kv going to share
  head_dim: int = n_embd // n_head         # What's the dim of each head?
  n_layer: int = 6 #32                         # how many layer?
  window_size: int = 4 #4096                     # if we use SWA then each token q only attn to tokens k,v that are in window.

  dropout: float = 0.2
  bias: bool = True

  # for selecting which layer to use
  lora: bool = False
  quant: bool = False                         # if wanna use pretrained quant(compressed weights) weights
