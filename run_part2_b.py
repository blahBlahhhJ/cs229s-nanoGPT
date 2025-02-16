import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import json
import argparse
from model import GPT
from utils import L2PruningHandler
import tiktoken
from tqdm import tqdm

# -----------------------------------------------------------------------------
prune_method = 'individual' # 'l2norm' or 'individual'

# -----------------------------------------------------------------------------
batch_size = 8
block_size = 1024
target_sparsity = 0.1
sparsity_increment = 0.01
target_loss = 3.2
real_data = True
seed = 1337
gradient_accumulation_steps = 40
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

curr_time = time.strftime("%m%d-%H%M%S")
log_file = open(f'part2b_{prune_method}_{curr_time}.log', 'a')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if real_data:
    dataset = 'wikitext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    start = 0
    def get_batch(split="train"):
        global start
        if split == "train":
            data = train_data
        elif split == "val":
            data = val_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if split == "train":
            ix = torch.randint(len(data) - block_size, (batch_size,))
        elif split == "val":
            ix = torch.arange(start, start + batch_size * block_size, block_size)
            start += batch_size * block_size
        else:
            raise ValueError(f"Invalid split: {split}")

        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

model = GPT.from_pretrained("gpt2-medium")

if prune_method == 'l2norm':
    handler = L2PruningHandler(model)
    handler.handle()

model.to(device)
optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
val_steps = (len(val_data)-1) // (batch_size * block_size) - 1

def train_model(model, iterations):
    model.train()
    train_loss = []
    for _ in tqdm(range(iterations), desc='training'):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            X, Y = get_batch('train')
            with ctx:
                _, loss = model(X, Y)
            loss.backward()
            train_loss.append(loss.item())
        model.prune_grad(method=prune_method)
        optimizer.step()
    lossf = np.mean(train_loss)
    return lossf

def get_loss(model, split='val'):
    model.eval()
    val_loss = []
    if split == 'val':
        steps = val_steps
    else:
        steps = 100
    with torch.no_grad():
        global start
        start = 0
        for _ in range(steps):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            val_loss.append(loss.item())
    val_lossf = np.mean(val_loss)
    model.train()
    return val_lossf

torch.cuda.synchronize()
# step 1: train 50 iterations
print("step 1: train 50 iterations", file=log_file, flush=True)
lossf = train_model(model, 50)
print(f"step 1 training loss: {lossf:.4f}", file=log_file, flush=True)
# validation
val_lossf = get_loss(model)
print(f"step 1 validation loss: {val_lossf:.4f}", file=log_file, flush=True)

torch.cuda.synchronize()
# step 2: iteratively prune and retrain according to validation loss
print("step 2: iterative pruning and retraining", file=log_file, flush=True)
for sparsity in np.arange(1.0-sparsity_increment, 0.0, -sparsity_increment):
    # floating point issue when subtracting sparsity_increment
    if round(sparsity,2) < target_sparsity:
        print(f"current sparsity {sparsity:.2f} is less than target sparsity {target_sparsity:.2f}, stop pruning", file=log_file, flush=True)
        break
    model.prune_weight(sparsity=sparsity, method=prune_method)
    lossf = get_loss(model, split='train')
    print(f"current sparsity {sparsity:.2f} train loss: {lossf:.4f}", file=log_file, flush=True)
    if lossf >= target_loss:
        print("finetune model", file=log_file, flush=True)
        lossf = train_model(model, 25)
        print(f"current sparsity {sparsity:.2f} training loss after finetuning: {lossf:.4f}", file=log_file, flush=True)

log_file.close()