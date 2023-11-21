"""
Debug inference
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPT
import tiktoken

# -----------------------------------------------------------------------------
init_from = 'gpt2-medium'
batch_size = 12
block_size = 1024
bias = False
real_data = True
dataset = 'shakespeare'
seed = 1337
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

for i in range(1):
    model.generate(ctx, 128, 0.4, 200)

t = time.time()
for i in range(3):
    torch.manual_seed(i + seed)
    y = model.generate(ctx, 128, 0.4, 200)
fp32t = time.time() - t
print(enc.decode(y[0].tolist()))
print('-----------------------------------')

# cast to fp16/bf16
model.to(ptdtype)

t = time.time()
for i in range(3):
    torch.manual_seed(i + seed)
    y = model.generate(ctx, 128, 0.4, 200)
bf16t = time.time() - t
print(enc.decode(y[0].tolist()))
print('-----------------------------------')

# quantize to uint8
model.quantize()

t = time.time()
for i in range(3):
    torch.manual_seed(i + seed)
    y = model.generate(ctx, 128, 0.4, 200)
uint8t = time.time() - t
print(enc.decode(y[0].tolist()))
print('-----------------------------------')

print('fp32:', fp32t)
print('bf16:', bf16t)
print('uint8:', uint8t)



# if profile:
#     # useful docs on pytorch profiler:
#     # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
#     # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
#     wait, warmup, active = 5, 5, 5
#     num_steps = wait + warmup + active
#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
#         record_shapes=False,
#         profile_memory=False,
#         with_stack=False, # incurs an additional overhead, disable if not needed
#         with_flops=True,
#         with_modules=False, # only for torchscript models atm
#     ) as prof:

#         X, Y = get_batch('train')
#         for k in range(num_steps):
#             with ctx:
#                 logits, loss = model(X, Y)
#             X, Y = get_batch('train')
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             lossf = loss.item()
#             print(f"{k}/{num_steps} loss: {lossf:.4f}")

#             prof.step() # notify the profiler at end of each step

# else:

#     # simple benchmarking
#     torch.cuda.synchronize()
#     for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
#         t0 = time.time()
#         X, Y = get_batch('train')
#         for k in range(num_steps):
#             with ctx:
#                 logits, loss = model(X, Y)
#             X, Y = get_batch('train')
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             lossf = loss.item()
#             print(f"{k}/{num_steps} loss: {lossf:.4f}")
#         torch.cuda.synchronize()
#         t1 = time.time()
#         dt = t1-t0
#         mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
#         if stage == 1:
#             print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
