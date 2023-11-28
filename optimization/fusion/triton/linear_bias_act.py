import math
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

import triton
import triton.language as tl


def get_configs(block_k):
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": block_k},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": block_k},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": block_k},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": block_k},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": block_k},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": block_k},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": block_k},
            num_stages=3,
            num_warps=4,
        ),
    ]

_kAlpha = math.sqrt(2.0 / math.pi)

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))

@triton.jit
def gelu_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)

# fmt: off
@triton.autotune(
    configs=[c for block_k in [32, 64] for c in get_configs(block_k)],
    key=["M", "N", "K"],
)
@triton.heuristics({
    'EVEN_N': lambda args: args["N"] % (args['BLOCK_N']) == 0,
})
@triton.jit
def kernel_fw(
    # Pointers to matrices
    OUT, ACT_INPUTS, INPUT, WEIGHT, bias,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_om, stride_im,
    stride_wn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    BIAS: tl.constexpr,
    SAVE_ACT_INPUTS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    is_fp16: tl.constexpr,  # autotune
):
    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """

    # programs are grouped together to improve L2 hit rate
    # the logic is that we'll consolidate over K. If the programs were not grouped,
    # then multiple cols/rows in the result would end up pulling in the same row and lines
    # from the inputs. By grouping the computation we ensure some data reuse, which the hardware
    # covers via the L2 cache
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_N)  # number of programs ids along the N axis
    num_pid_in_group = GROUP_M * num_pid_n  # number of programs in group
    group_id = pid // num_pid_in_group  # id of the group this program is in
    first_pid_m = group_id * GROUP_M  # row-id of the first program in the group
    GROUP_M = min(
        num_pid_m - first_pid_m, GROUP_M
    )  # if `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller

    # *within groups*, programs are ordered in a column-major order
    # row-id /col-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # the memory addresses of elements can follow numpy broadcasting
    input_ptrs = INPUT + rm[:, None] * stride_im
    weight_ptrs = WEIGHT + rn[None, :] * stride_wn

    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if BIAS:
        if EVEN_N:
            bias = tl.load(bias + rn).to(tl.float32)
        else:
            bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # block level matrix multiplication.
    # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    mask_rn = rn < N
    mask_rm = rm < M

    for i in range(0, K, BLOCK_K):
        rk = tl.arange(0, BLOCK_K) + i
        a = tl.load(input_ptrs + rk[None, :], mask=((rk[None, :] < K) & mask_rm[:, None]), other=0.0)
        w = tl.load(weight_ptrs + rk[:, None], mask=((rk[:, None] < K) & mask_rn[None, :]), other=0.0)

        acc += tl.dot(a, w)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # optional: save the activation inputs
    if SAVE_ACT_INPUTS:
        act_in_ptrs = ACT_INPUTS + rm[:, None] * stride_om + rn[None, :]
        tl.store(act_in_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == 1:
        acc = gelu(acc)

    # write back result
    out_ptrs = OUT + rm[:, None] * stride_om + rn[None, :]
    tl.store(out_ptrs, acc, mask=mask_rm[:, None] & mask_rn[None, :])

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_stages=3, num_warps=4),
    ],
    key=["N"],
)
@triton.heuristics({
    'EVEN_N': lambda args: args["N"] % (args['BLOCK_N']) == 0,
})
@triton.jit
def kernel_bw(
    # Pointers to matrices
    GRAD_ACT, GRAD_OUT, ACT_INPUTS,
    # Matrix dimensions
    N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_gom, stride_aim,
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ACTIVATION_GRAD: tl.constexpr,
):
    # fmt: on

    """
    Go over all the activation inputs, compute the corresponding gradient
    """

    # this kernel is relatively simple in terms of scheduling:
    # - per row (pid_m)
    # - each program a given chunk on the col axis,
    # since it's more effective memory and occupancy wise
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # the memory addresses of elements in the first block of
    # A and W can be computed using numpy-style broadcasting
    act_input_ptrs = ACT_INPUTS + pid_m * stride_aim + rn

    # compute the gradient which is related to this activation
    if EVEN_N:
        act_in = tl.load(act_input_ptrs)
    else:
        act_in = tl.load(act_input_ptrs, mask=rn < N, other=0.0)

    if ACTIVATION_GRAD == 1:
        grad_act = gelu_grad(act_in)
    else:
        grad_act = act_in

    # now read the incoming gradient, the backpropagated one is the multiple of both
    grad_out_ptrs = GRAD_OUT + pid_m * stride_gom + rn
    if EVEN_N:
        grad_out = tl.load(grad_out_ptrs)
    else:
        grad_out = tl.load(grad_out_ptrs, mask=rn < N)

    grad_act *= grad_out

    # write back result
    grad_act_ptrs = GRAD_ACT + pid_m * stride_gom + rn
    tl.store(grad_act_ptrs, grad_act, mask=rn < N)


class _linear_bias_act(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        x,
        weight,
        bias,
        activation,
        trainable_weight,
        trainable_bias,
        save_activation_inputs,
    ):

        if not x.is_contiguous():
            x = x.contiguous()

        x_ = x if x.ndim == 2 else x.flatten(0, -2)

        assert (
            x_.shape[1] == weight.shape[1]
        ), f"Incompatible dimensions in between inputs and weight, {x_.shape} - {weight.shape}"
        assert bias is None or bias.is_contiguous()
        assert (
            bias is None or bias.shape[0] == weight.shape[0]
        ), "Incompatible dimensions in between weight and bias"
        assert weight.is_contiguous()

        M, K = x_.shape
        N, K = weight.shape

        outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)
        activation_inputs = torch.empty_like(outputs) if save_activation_inputs else x  # will not be used in that case

        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),) # noqa

        kernel_fw[grid](
            outputs, activation_inputs, x_, weight,            # data ptrs
            bias if bias is not None else x,            # auto skip bias if not present
            M, N, K,                                    # shapes
            outputs.stride(0), x_.stride(0),            # strides
            weight.stride(0),
            ACTIVATION=activation,                      # optional fused activation
            BIAS=bias is not None,                      # optional fused bias
            GROUP_M=8,                                  # speed optimization: group the programs
            SAVE_ACT_INPUTS=save_activation_inputs,
            is_fp16=x_.dtype == torch.float16
        )

        activation_inputs = activation_inputs if save_activation_inputs else None

        outputs = outputs if x.ndim == 2 else outputs.reshape(*x.shape[:-1], N)

        ctx.activation = activation
        ctx.trainable_weight = trainable_weight
        ctx.trainable_bias = trainable_bias

        # Micro-optimization: saving these is not always needed (?)
        if x.requires_grad or ctx.trainable_weight or ctx.trainable_bias:
            ctx.save_for_backward(weight, activation_inputs, x)

        return outputs

    @staticmethod
    @custom_bwd
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> Any:
        """
        Compute the derivative with respect to x, other tensors were not trainable inputs.
        """
        (weight, activation_inputs, inputs) = ctx.saved_tensors
        trainable_weight = ctx.trainable_weight
        trainable_bias = ctx.trainable_bias
        activation_grad = ctx.activation

        # Make sure that we don't have to handle the stride over cols
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        grad_out_ = grad_out if grad_out.ndim == 2 else grad_out.flatten(0, -2)
        inputs_ = inputs if inputs.ndim == 2 else inputs.flatten(0, -2)

        assert grad_out_.shape[1] == weight.shape[0], "Incompatible dimensions in between grad_out and weight"

        M, N = grad_out_.shape
        N, _ = weight.shape

        # Compute the gradient for the activation
        if activation_grad > 0:
            grad_act = torch.empty_like(grad_out_)

            # Some activations do not require their inputs to
            # know of their grad, the downstream grad is enough
            if activation_inputs is None:
                activation_inputs = grad_out_

            grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"])) # noqa

            kernel_bw[grid](
                grad_act, grad_out_, activation_inputs,            # data ptrs
                N,                                      # shapes
                grad_act.stride(0), activation_inputs.stride(0),   # strides
                ACTIVATION_GRAD=activation_grad,        # optional fused activation
            )

            # Backpropagation going up, the reference gradient is now
            # just before the activation
            grad_out_ = grad_act

        # The following ops can also be handled by pytorch
        grad_in = triton.ops.matmul(grad_out_, weight)
        grad_weight = grad_out_.transpose(1, 0) @ inputs_ if trainable_weight else None
        grad_bias = torch.sum(grad_out_, dim=0) if trainable_bias else None

        return grad_in.reshape_as(inputs), grad_weight, grad_bias, None, None, None, None, None, None
    
def linear_bias_act(
    x,
    weight,
    bias,
    activation=0,
    trainable_weight=True,
    trainable_bias=True,
    save_activation_inputs=True,
):
    # print(x.shape, weight.shape, bias.shape)
    return _linear_bias_act.apply(
        x,
        weight,
        bias, 
        activation, 
        trainable_weight, 
        trainable_bias, 
        save_activation_inputs,
    )