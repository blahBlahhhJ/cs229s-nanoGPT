"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import json
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
result = dict()

# -----------------------------------------------------------------------------
batch_size = 8
block_size = 1024
bias = False
real_data = True
seed = 1337
gradient_accumulation_steps = 40
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
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

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

#---------------------------------------------------------------------------------
# First task: train the model and get validation loss

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 500
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([2, 500]): # burnin, then benchmark
        t0 = time.time()
        for k in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(gradient_accumulation_steps):
                X, Y = get_batch('train')
                with ctx:
                    logits, loss = model(X, Y)
                loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")

# validation loss
val_steps = (len(val_data)-1) // (batch_size * block_size) - 1
batch_loss = []
with torch.no_grad():
    for k in range(val_steps):
        X, Y = get_batch('val')
        with ctx:
            logits, loss = model(X, Y)
        lossf = loss.item()
        print(f"{k}/{val_steps} validation loss: {lossf:.4f}")
        batch_loss.append(lossf)
val_loss = np.mean(batch_loss)
print(f"Validation loss: {val_loss:.4f}")
result['loss'] = val_loss

# ------------------------------------------------------------------------------------
# Second task: inference latency







# -----------------------------------------------------------------------------------
with open('result.json', 'w') as f:
    json.dump(result, f)