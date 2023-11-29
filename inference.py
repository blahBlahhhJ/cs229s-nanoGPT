"""
Debug inference
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from optimization.model import GPT
import tiktoken

# -----------------------------------------------------------------------------
init_from = 'gpt2-medium'
batch_size = 12
block_size = 1024
bias = False
real_data = True
dataset = 'shakespeare'
seed = 1337
num_samples = 3
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
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

# import pdb; pdb.set_trace()
# model init
model = GPT.from_pretrained(init_from)
model.eval()
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

enc = tiktoken.get_encoding("gpt2")
ctx = torch.tensor(enc.encode("hello everyone, my name is", allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device).unsqueeze(0)
ctx = ctx.repeat(12, 1)

def benchmark(model, ctx):
    # warmup
    for i in range(1):
        model.generate_kv(ctx, 128, 0.4, 200)

    # torch.cuda.synchronize(device=None)
    # t = time.time()
    # for i in range(num_samples):
    #     torch.manual_seed(i + seed)
    #     y = model.generate(ctx, 128, 0.4, 200)
    # fp32t = time.time() - t
    # print(enc.decode(y[0].tolist()))
    # print('-----------------------------------')

    torch.cuda.synchronize(device=None)
    t = time.time()
    for i in range(num_samples):
        torch.manual_seed(i + seed)
        y = model.generate_kv(ctx, 128, 0.4, 200)
    fp32t_kv = time.time() - t
    print(enc.decode(y[0].tolist()))
    print('-----------------------------------')

    model.to(ptdtype)

    # torch.cuda.synchronize(device=None)
    # t = time.time()
    # for i in range(num_samples):
    #     torch.manual_seed(i + seed)
    #     y = model.generate(ctx, 128, 0.4, 200)
    # fp16t = time.time() - t
    # print(enc.decode(y[0].tolist()))
    # print('-----------------------------------')

    torch.cuda.synchronize(device=None)
    t = time.time()
    for i in range(num_samples):
        torch.manual_seed(i + seed)
        y = model.generate_kv(ctx, 128, 0.4, 200)
    fp16t_kv = time.time() - t
    print(enc.decode(y[0].tolist()))
    print('-----------------------------------')

    # quantize to uint8
    # model.quantize()

    # torch.cuda.synchronize(device=None)
    # t = time.time()
    # for i in range(num_samples):
    #     torch.manual_seed(i + seed)
    #     y = model.generate(ctx, 128, 0.4, 200)
    # uint8t = time.time() - t
    # print(enc.decode(y[0].tolist()))
    # print('-----------------------------------')

    # print('fp32:', fp32t/num_samples)
    print('fp32_kv:', fp32t_kv/num_samples)
    # print('fp16:', fp16t/num_samples)
    print('fp16_kv:', fp16t_kv/num_samples)
    # print('uint8:', uint8t/num_samples)
    pass

benchmark(model, ctx)

# wait, warmup, active = 1, 1, 1
# num_steps = wait + warmup + active
# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log/bdr'),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=False, # incurs an additional overhead, disable if not needed
#     with_flops=True,
#     with_modules=False, # only for torchscript models atm
# ) as prof:

#     for i in range(num_steps):
#         torch.manual_seed(i + seed)
#         y = model.generate_kv(ctx, 128, 0.4, 200)

#         prof.step() # notify the profiler at end of each step
