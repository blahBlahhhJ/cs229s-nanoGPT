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
batch_size = 1
block_size = 1024
num_samples = 5
generate_num = 50
topk = 1
seed = 1337
gradient_accumulation_steps = 40
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

curr_time = time.strftime("%m%d-%H%M%S")
log_file = open(f'part1_decoding_{curr_time}.log', 'a')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = GPT.from_pretrained("gpt2-medium")
# model.to(torch.bfloat16)
model.to(device)

draft_model = GPT.from_pretrained("gpt2")
draft_model.enable_kv()
draft_model.to(torch.bfloat16)
draft_model.to(device)

print("standard vs speculative decoding with main model gpt2-medium and draft model gpt2-small", file=log_file, flush=True)

enc = tiktoken.get_encoding("gpt2")
dataset = 'wikitext'
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
i = 229
x = torch.from_numpy((val_data[i:i+block_size]).astype(np.int64))
ctx = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
# ctx = torch.tensor(enc.encode("Hello my name is ", allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device).unsqueeze(0)
ctx = ctx.repeat(batch_size, 1)

print("------------------- warm up ---------------------")
for i in range(1):
    model.generate(ctx, generate_num, 0.4, topk)
print('-------------------------------------------------')

torch.cuda.synchronize(device=None)
t = time.time()
for i in range(num_samples):
    torch.manual_seed(i + seed)
    y = model.generate(ctx, generate_num, 0.4, topk)
fp32t_kv = time.time() - t
# print(enc.decode(y[0].tolist()))
print('-------------------------------------------------')
print('standard decoding latency (seconds):', fp32t_kv/num_samples, file=log_file, flush=True)

print("------------------- warm up ---------------------")
for i in range(1):
    model.generate_speculative(ctx, generate_num, draft_model, 0.4, topk)
print('-------------------------------------------------')

torch.cuda.synchronize(device=None)
t = time.time()
for i in range(num_samples):
    torch.manual_seed(i + seed)
    y = model.generate_speculative(ctx, generate_num, draft_model, 0.4, topk)
fp32t_kv = time.time() - t
# print(enc.decode(y[0].tolist()))
print('-------------------------------------------------')
print('speculative decoding latency (seconds):', fp32t_kv/num_samples, file=log_file, flush=True)

print("standard vs speculative decoding with main model gpt2-medium and draft model gpt2-medium quantized", file=log_file, flush=True)

print("------------------- warm up ---------------------")
for i in range(1):
    model.generate(ctx, generate_num, 0.4, topk)
print('-------------------------------------------------')

torch.cuda.synchronize(device=None)
t = time.time()
for i in range(num_samples):
    torch.manual_seed(i + seed)
    y = model.generate(ctx, generate_num, 0.4, topk)
fp32t_kv = time.time() - t
print(enc.decode(y[0].tolist()))
print('-------------------------------------------------')
print('standard decoding latency (seconds):', fp32t_kv/num_samples, file=log_file, flush=True)

draft_model = GPT.from_pretrained("gpt2-medium")
draft_model.enable_kv()
draft_model.to(ptdtype)
handler = WeightOnly8BitHandler(draft_model)
handler.handle()
draft_model.to(device)

print("------------------- warm up ---------------------")
for i in range(1):
    model.generate_speculative(ctx, generate_num, draft_model, 0.4, topk)
print('-------------------------------------------------')

torch.cuda.synchronize(device=None)
t = time.time()
for i in range(num_samples):
    torch.manual_seed(i + seed)
    y = model.generate_speculative(ctx, generate_num, draft_model, 0.4, topk)
fp32t_kv = time.time() - t
# print(enc.decode(y[0].tolist()))
print('-------------------------------------------------')
print('speculative decoding latency (seconds):', fp32t_kv/num_samples, file=log_file, flush=True)