# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Any, Dict, Optional

import torch
import triton

from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print

from optimization.fusion.functional import fused_bias_dropout_residual

SHAPES = [
    (1, 1, 4096),
]

P = 0.1


def to_gbs_fw(a, ms, bias):
    # Read and write the full array
    total = 2 * a.numel() * a.element_size()

    if bias:
        # Read the bias, ideally only once
        total += a.shape[-1] * a.element_size()

    return total * 1e-9 / (ms * 1e-3)


def bench_dropout(bias: bool, backward: bool):
    device = torch.device("cuda")

    for dtype in [
        torch.float16,
        torch.float32,
    ]:
        results: Dict[str, Any] = {}

        for B, M, K in SHAPES:
            a = torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=backward
            )
            r = torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=backward
            )
            b = torch.rand(K, device=device, dtype=dtype, requires_grad=backward)

            def triton_dropout(x):
                return fused_bias_dropout_residual(x, P, r, backward, b if bias else None)
            
            def torch_dropout(x, bias: bool = bias, b: torch.Tensor = b, P: float = P, r: torch.Tensor = r):
                if bias:
                    x = x + b
                    out = torch.nn.functional.dropout(x, p=P, training=backward)
                    out = r + out
                    return out
                else:
                    out = torch.nn.functional.dropout(x, p=P, training=backward)
                    out = r + out
                    return out


            def torch_step(x):
                y = torch_dropout(x)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y

            def triton_step(x):
                y = triton_dropout(x)
                if backward:
                    y.grad = None
                    torch.norm(y).backward()
                return y

            for testcase in [
                TestCase(
                    torch_step,
                    "pytorch - bias: {} - fw{}".format(
                        bias, "+bw" if backward else ""
                    ),
                ),
                TestCase(
                    triton_step,
                    "triton - bias: {} - fw{}".format(
                        bias, "+bw" if backward else ""
                    ),
                ),
            ]:
                time = triton.testing.do_bench(
                    lambda: testcase.function(a), grad_to_none=[a, b]
                )
                key = f"B={B}, M={M}, K={K}"
                if key not in results:
                    results[key] = {}

                # Record BW
                bandwidth = to_gbs_fw(a, time, bias)
                results[key][testcase.name] = f"{bandwidth:.1f}"

        pretty_print(results, title="\n --- Type: {} --- ".format(dtype), units="GB/s")
        # pretty_plot(
        #     results,
        #     title="Dropout-Bias-{}-FW{}-{}".format(
        #         bias, "+BW" if backward else "", dtype
        #     ),
        #     units="GB/s",
        #     dash_key="pytorch",
        # )


for bw in [True, False]:
    for bias in [True]:
        bench_dropout(bias, bw)