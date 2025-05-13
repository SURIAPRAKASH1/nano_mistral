from dataclasses import dataclass
import torch 

@dataclass
class MistralConfig:

  n_embd: int = 192                             # token embedding (just a vector representation haahaa) dim
  vocab_size:int = 32000                        # total unique tokens
  block_size: int = 100                         # just a seq_len.
  batch_size: int = 32                          # How may block (seq) are packed togather
  n_head: int = 8                               # How many heads?
  n_kv_head: int = 2                            # if we use GQA then kv going to share
  head_dim: int = n_embd // n_head              # What's the dim of each head?
  n_layer: int = 6                              # How many layer?
  window_size: int = block_size // 2         # if we use SWA then each token's q only attn to token's k,v only in window.
  hidden_dim: int = int(n_embd * 3.5)           # mlp or moe intermediate dim

  dropout: float = 0.2
  bias: bool = False

  # for selecting which linear layer to use
  lora: bool = False
  quant: bool = False                         # if we wanna use pretrained quant(compressed weights) weights
  dtype: torch.TensorType = torch.float16 if quant else torch.get_default_dtype()

  # rope stuffs
  omega: int = 10 ** 7
  max_block_size: int = block_size
