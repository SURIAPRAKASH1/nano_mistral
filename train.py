import importlib.util
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer  

from collections import defaultdict
import importlib
import matplotlib.pyplot as plt


from model.config import MistralConfig
from model.transformer import MistralTransformer

# what's the current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device -->', device)

# tokenizer
if importlib.util.find_spec('transformers'):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    vocab_size = tokenizer.vocab_size
    print('vocab_size', vocab_size)
else:
    print('we need mistral tokenizer from transformers lib so install !') 
    
# lyrics file 
text = open("All_eminem_songs.txt", 'r').read()
tokens = tokenizer.encode(text)
print(f'{len(text)} words get tokenized to {len(tokens)} tokens')


# splitting data into train and dev set's
n = int(len(tokens) * 0.9)
train_data = tokens[:n]      # 90%
dev_data = tokens[n:]        # 10%
print(f"train data {len(train_data)} tokens\ndev data {len(dev_data)} tokens")


# randomly get sample of batch of tokens
def get_batch(split, device):
  data = train_data if split == 'train' else dev_data
  xi = torch.randint(len(data) - MistralConfig.block_size, (MistralConfig.batch_size,))
  x = torch.tensor([data[i: i + MistralConfig.block_size] for i in xi])
  y = torch.tensor([data[i + 1: i + MistralConfig.block_size + 1 ] for i in xi])

  # for efficient gpu performence
  if device != 'cpu':
    x = x.pin_memory()              # by pinning make sure tensor ain't non pageble (only live in ram)
    y = y.pin_memory()
    x = x.to(device, non_blocking = True)
    y = y.to(device, non_blocking = True)
  else:
    x = x.to(device)
    y = y.to(device)

  return x, y


X, Y = get_batch('train', device)
# How transfomer see tokens and learn from it
# for single sequence . here i cut the seq for visualization
for i in range(MistralConfig.block_size // 10):
  t_input = X[0, : i+1].tolist()
  t_pred = Y[0, i].tolist()
  print("-----------------------------------HOW TRANSFORMER SEE TOKENS AND LEARN FROM IT---------------------------------")
  print(f"Input: {t_input}, Have to predict: {t_pred}")
  print(f"Input: {tokenizer.decode(t_input)}, Have to predict: {tokenizer.decode(t_pred)}")
  print(' ') 


# Hyper parameters for training
steps = 5000           # how may steps we want to trian our model
eval_iters = 200       # when estimating loss how many batches we should be consider
eval_step = 500        # for printing loss
lr = 6e-4     
min_lr = 6e-5

weight_decay = 1e-3
warmup_iters = 400    # will increase lr then start to decay from here 

@torch.no_grad()
def estimate_loss(model):
  model.eval()              # model in eval mode bro .....

  losses = {}
  for split in ['train', 'dev']:
    l = []
    for _ in range(eval_iters):
      X, Y = get_batch(split, device = device)
      _, loss = model(X, Y)
      l.append(loss)
    # take average over batches
    losses[split] = torch.stack(l, dim = 0).mean(0)

  model.train()
  return losses

import math
def get_lr(it):

  # so we gradually increasing learning rate
  if it < warmup_iters:
    return  lr * (it + 1) / (warmup_iters + 1)

  # starting to decaying the learning rate using cosine
  else:
    decay_ratio = (it - warmup_iters)/ (steps - warmup_iters)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * ( 1.0 + math.cos( math.pi * decay_ratio))
    return  min_lr + coeff * (lr - min_lr)    # we make sure learning rate shouldn't 0 (but we wanna decrease)
  

model = MistralTransformer(MistralConfig).to(device)

# AdamW (decoubled weight decay)
optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay= weight_decay)
scaler = torch.amp.GradScaler(device = device)

# loss stacks
gb_lossi = defaultdict(list)

# Optimization loop
for step in range(steps):
  # get batch of sample data from training dataset
  X, Y = get_batch('train', device)
  optimizer.zero_grad()

  # 1. FORWARD PASS AND COMPUTE LOSS

  # enable auto mixed percision. it's converts dtype to F16/BF16 whenever possible.
  with torch.amp.autocast(device_type= device):
    _, loss = model(X, Y)

  # 2. BAKWARD PASS
  # scale the loss then do back-ward pass
  # cause computing loss in F16 dtype (if we) we get very small loss. if compute grad for that we will get vanishing gradients
  # so what's the solution scale the loss then compute gradients, when updating params scale down else explode
  scaler.scale(loss).backward()

  # grad clip
  scaler.unscale_(optimizer) 
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

  # 3.UPDATE PARAMETERS 
  scaler.step(optimizer)
  scaler.update()
  # lr_scheduler.step(step + 1/ steps)
  optimizer.defaults['lr'] = get_lr(step) 

  # estimate loss once in a while
  if step % eval_step == 0 or step == steps - 1:
    losses = estimate_loss(model)

    gb_lossi['train'].append(losses['train'].item())
    gb_lossi['dev'].append(losses['dev'].item())

    print(f"{step}:{steps}, train_loss: {losses['train'].item()}, dev_loss: {losses['dev'].item()} ")


# visualizing loss
plt.plot(gb_lossi['train'])
plt.plot(gb_lossi['dev']) 
plt.xlabel(f'per {eval_iters}/Batches')
plt.ylabel('Loss')
plt.legend(['train', 'dev'])
plt.show()


# SAMPLING 

# encode string to get tokens
prompt = """
look if you had one shot one oppurtunity 
to sieze everything you ever wanted one moment 
"""
encoded_tokens =  torch.tensor([tokenizer.encode(prompt)], device= device) # (B, T) 

# sampling from model
model.eval()
generated_tokens =  model.generate(encoded_tokens, max_tokens= 100, temprature= 0.7, top_k= 200, top_p= 0.8)

# decode tokens to get string format 
result = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens= True)
print(result)
