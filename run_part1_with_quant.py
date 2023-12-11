import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import json
from model import GPT
from optimization import WeightOnly8BitHandler
import tiktoken
from tqdm import tqdm

# -----------------------------------------------------------------------------
result = dict()

# -----------------------------------------------------------------------------
batch_size = 4
block_size = 1024
bias = False
real_data = True
seed = 1337
gradient_accumulation_steps = 40
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

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
            ix = torch.arange(start, start + batch_size * block_size, block_size)
            start += batch_size * block_size
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

print("model with quantization")
model.eval()
model.to(ptdtype)
handler = WeightOnly8BitHandler(model)
handler.handle()
model.to(device)
print("max GPU memory allocated:", torch.cuda.max_memory_allocated())
for i, data in enumerate([train_data, val_data]):
    if i == 0:
        continue
    # reset indicator
    start = 0
    # len(data)-1 because the label needs to be shifted by 1
    train_or_val = "train" if i == 0 else "val"
    steps = (len(data)-1) // (batch_size * block_size) - 1
    batch_loss = []
    with torch.no_grad():
        for k in tqdm(range(steps)):
            X, Y = get_batch(train_or_val)
            with ctx:
                _, loss = model(X, Y)
            lossf = loss.item()
            # print(f"{k}/{steps} {train_or_val} loss: {lossf:.4f}")
            batch_loss.append(lossf)
    loss = np.mean(batch_loss)
    print("max GPU memory allocated after inference:", torch.cuda.max_memory_allocated())
    print(f"{train_or_val} loss: {loss:.4f}")
    print(f"{train_or_val} perplexity: {np.exp(loss):.4f}")


