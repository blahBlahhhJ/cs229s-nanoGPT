import math
from typing import Any, Optional

import torch
import torch.nn as nn

import triton
import triton.language as tl


BLOCK_M = 1
BLOCK_N = 1024  # NOTE: This should ideally be GPU dependent, big impact on perf


_configs = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
]


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda args: args["BLOCK_N"] * args["BLOCK_M"]})
@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def kernel_fw(
    Y, X, BIAS, SEEDS, RESIDUAL,
    stride,
    M, N,
    p: tl.constexpr,
    is_fp16: tl.constexpr,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    USE_BIAS: tl.constexpr,
    TRAINING: tl.constexpr,
):
    """
    Apply dropout on an input tensor
    Y : Output  (M, N)
    X : Input   (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    # fmt: on

    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    x_ptrs = X + rows[:, None] * stride + cols[None, :]
    y_ptrs = Y + rows[:, None] * stride + cols[None, :]
    r_ptrs = RESIDUAL + rows[:, None] * stride + cols[None, :]

    # good to go, start the layer computations
    col_mask = cols[None, :] < N
    p_scale = 1. / (1. - p)
    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.)
    else:
        bias = x_ptrs  # will not be used

    block_mask = (rows[:, None] < M) & col_mask
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)
    residual = tl.load(r_ptrs, mask=block_mask, other=0.0)

    # optionally apply a fused bias
    if USE_BIAS:
        if TRAINING:
            x += bias
        else:
            x += bias + residual

    if TRAINING:
        # get the random keep mask
        rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
        seed_int = tl.load(SEEDS + col_id)
        r = tl.rand(seed_int, rand_offsets)
        keep_mask = r > p

        # prune and normalize in one go
        keep = tl.view(keep_mask, x.shape)
        x = tl.where(keep, (x * p_scale).to(x.dtype), 0.)
        output = x + residual
    else:
        output = x

    tl.store(y_ptrs, output, mask=block_mask)  # output


# fmt: off
@triton.heuristics({"SIZE_RAND_BLOCK": lambda args: args["BLOCK_N"] * args["BLOCK_M"]})
@triton.autotune(
    configs=_configs,
    key=["M", "N", "is_fp16"],
)
@triton.jit
def kernel_bw(
    GRAD_IN, GRAD_BIAS, GRAD_OUT,
    BIAS, SEEDS,
    stride_grad, stride_inputs,
    M, N,
    p: tl.constexpr,
    is_fp16: tl.constexpr,  # autotune
    # Meta-parameters
    BLOCK_M: tl.constexpr,  # heuristics
    BLOCK_N: tl.constexpr,
    SIZE_RAND_BLOCK: tl.constexpr,
    TRAINABLE_BIAS: tl.constexpr,
    USE_BIAS: tl.constexpr,
    TRAINING: tl.constexpr,
):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_BIAS   (N,)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (N,)
    p : dropout probability
    """
    # fmt: on

    row_id = tl.program_id(axis=0)
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers starting point
    grad_out_ptrs = GRAD_OUT + rows[:, None] * stride_grad + cols[None, :]
    grad_in_ptrs = GRAD_IN + rows[:, None] * stride_grad + cols[None, :]

    # now go over the tiles
    grad_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    col_mask = cols[None, :] < N
    p_scale = 1. / (1. - p)

    if USE_BIAS:
        b_ptrs = BIAS + cols[None, :]
        bias = tl.load(b_ptrs, mask=col_mask, other=0.)

    block_mask = (rows[:, None] < M) & col_mask
    grad_out = tl.load(grad_out_ptrs, mask=block_mask, other=0.)

    if TRAINING:
        # randomly prune (and scale) the resulting buffer, possibly a no-op
        # note that even if we did not save the mask from the FW pass, it is generated
        # from the same seeds, so the same drop mask is applied here
        rand_offsets = tl.arange(0, SIZE_RAND_BLOCK)
        seed_int = tl.load(SEEDS + col_id)
        r = tl.rand(seed_int, rand_offsets)
        r = tl.view(r, grad_out.shape)
        output = tl.where(r > p, (grad_out * p_scale).to(grad_out.dtype), 0.)
    else:
        output = grad_out
    # write-back
    tl.store(grad_in_ptrs, output, mask=block_mask)

    # optionally accumulate the bias gradient
    if TRAINABLE_BIAS:
        grad_bias += tl.sum(output, axis=0)

    if TRAINABLE_BIAS:
        grad_bias_ptr = GRAD_BIAS + row_id * N + cols
        tl.store(grad_bias_ptr, grad_bias, mask=cols < N)


class _bias_dropout_residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, bias, residual, trainable_bias, training):
        # Soft-flatten an hypothetical 3rd dimension
        y = torch.empty_like(x)
        M, N = 1, x.shape[-1]
        for s in x.shape[:-1]:
            M *= s

        assert bias is None or (bias.dtype == x.dtype and bias.shape[0] == N)
        assert not training or p > 0.0

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
            )

        N_BLOCK_N = triton.cdiv(N, BLOCK_N)

        # Generate one seed per sample
        # seed max is int32 max for positive numbers: 2**16
        if training:
            seeds = torch.randint(65536, (N_BLOCK_N,), device=x.device, dtype=torch.int32)
        else:
            seeds = None

        # fmt: off
        bias_ptr = bias if bias is not None else x_  # Possibly not being used

        kernel_fw[grid](
            y, x,
            bias_ptr,
            seeds,
            residual,
            y.stride(0),
            M, N,
            p,
            x.dtype == torch.float16,
            USE_BIAS=bias is not None,
            TRAINING=training,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        # fmt: on

        ctx.save_for_backward(seeds, bias)

        ctx.trainable_bias = bias is not None and trainable_bias
        ctx.training = training
        ctx.p = p

        return y

    @staticmethod
    def backward(
        ctx, grad_out
    ):  # pragma: no cover  # This is covered, but called from C++ and not tracked
        (seeds, bias) = ctx.saved_tensors
        training = ctx.training

        # Soft-flatten an hypothetical 3rd dimension
        grad_out_ = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_in = torch.empty_like(grad_out_)

        M, N = grad_out_.shape

        inputs = grad_out_

        # We split the problem in tiles:
        # - over M there will be a follow up reduction
        # - over N we compromise in between trying to use as much memory paralellism as possible,
        # (fill in the warps, there are 32 threads per warps, and 4 warps default), and not being too
        # big because of register spilling
        N_BLOCKS_M = triton.cdiv(M, BLOCK_M)

        if ctx.trainable_bias:
            grad_bias = torch.empty(
                (
                    N_BLOCKS_M,
                    N,
                ),
                device=grad_in.device,
                dtype=grad_in.dtype,
            )

        else:
            grad_bias = grad_in  # will not be used

        def grid(meta):
            # NOTE: We use Triton Philox random number generator, which optimally generates 4 blocks for
            # a given seed and offsets. "BLOCK_M" here describes the size of one of these blocks
            # but we need to take this factor of 4 into account when scheduling all the kernels
            return (
                N_BLOCKS_M,
                triton.cdiv(N, meta["BLOCK_N"]),
            )

        # fmt: off
        kernel_bw[grid](
            grad_in, grad_bias, grad_out_,
            bias if bias is not None else inputs,
            seeds,
            grad_out_.stride(0), inputs.stride(0),
            M, N,
            ctx.p,
            grad_in.dtype == torch.float16,
            USE_BIAS=bias is not None,
            TRAINABLE_BIAS=ctx.trainable_bias,
            TRAINING=training,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        # fmt: on

        return (
            grad_in.reshape_as(grad_out),
            None,
            torch.sum(grad_bias, dim=0) if ctx.trainable_bias else None,
            grad_out,
            None,
            None,
            None,
            None,
        )


def bias_dropout_residual(
    x: torch.Tensor,
    p: float,
    residual: torch.Tensor,
    training: bool = True,
    bias: Optional[torch.Tensor] = None,
):
    """
    Apply dropout on the input tensor.
    Optionally add a bias, the computation will be fused.

    GPT: (1,1,1024) for x and residual.
    """

    assert p <= 1.0 and p >= 0.0

    if p == 1.0:
        return torch.zeros_like(x)

    # Micro optim, skip dropout
    # if p == 0.0:
    #     x = x + bias + residual if bias is not None else x + residual
    #     return x

    # The normal triton enabled codepath
    return _bias_dropout_residual.apply(
        x,
        float(p),
        bias,
        residual,
        bias is not None and bias.requires_grad,
        training,
    )
