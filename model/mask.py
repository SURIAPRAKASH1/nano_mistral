import torch

def swa_mask(block_size, window_size, device):
  """
  block_size : size of the seq_len
  window_size: how many previous tokens we can communicate
  device: device like object
  """

  i = torch.arange(block_size, device = device).unsqueeze(0) # row rep
  j = torch.arange(block_size, device = device).unsqueeze(1) # col rep

  # allowed positions
  mask = (i <= j) & (i >= j - window_size + 1)
  # convert to float and inf
  float_mask = torch.where(mask, torch.tensor(1.0, device = device), torch.tensor(0.0, device = device))
  return float_mask