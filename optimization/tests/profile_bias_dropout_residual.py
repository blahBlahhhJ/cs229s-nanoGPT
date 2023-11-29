import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Any, Dict, Optional

import torch

from optimization.fusion.layers import FusedLinearBiasDropoutResidual


SHAPES = [
    (12, 1, 1024, 1024),
    (12, 1, 4096, 1024),
]
P = 0.1
device = torch.device("cuda")
dtype = torch.float32
backward = False



wait, warmup, active = 5, 5, 5
num_steps = wait + warmup + active
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log/test'),
    record_shapes=True,
    profile_memory=True,
    with_stack=False, # incurs an additional overhead, disable if not needed
    with_flops=True,
    with_modules=False, # only for torchscript models atm
) as prof:
    for _ in range(num_steps):
        for B, M, k, K in SHAPES:
            l = FusedLinearBiasDropoutResidual(k, K, bias=True, device=device, dtype=dtype)
            x = torch.rand(
                (B, M, k), device=device, dtype=dtype, requires_grad=backward
            )
            r = torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=backward
            )

            x = l(x, r)

        prof.step()