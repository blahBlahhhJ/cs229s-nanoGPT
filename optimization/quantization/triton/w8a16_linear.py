"""
modified from https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/linear_layer.py
"""

from typing import Optional

import math
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


sqrt2 = math.sqrt(2.0)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k not used
                    # for split_k in [2, 4, 8, 16]:
                    #     configs.append(triton.Config(
                    #         {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #         num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / sqrt2))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
    ]
    + get_configs_io_bound(),
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
)
@triton.heuristics(
    {
        "K_LOAD_MASK_NEEDED": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _w16a16_linear_kernel(
    C,  # Pointers to matrices
    A,
    B,
    scale,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    output_m_stride,
    output_n_stride,
    a_m_stride,
    a_k_stride,
    b_n_stride,
    b_k_stride,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    K_LOAD_MASK_NEEDED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ACC_TYPE: tl.constexpr,
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
    program_idx = tl.program_id(axis=0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_idx = program_idx // width
    group_size = min(grid_m - group_idx * GROUP_M, GROUP_M)
    block_m_idx = group_idx * GROUP_M + (program_idx % group_size)
    block_n_idx = (program_idx % width) // group_size

    # now compute the block that each program will go through
    # m_offs (resp. n_offs) denotes a range of indices
    # for rows (resp. col) of C
    m_offs_untagged = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs_untagged = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # trick to avoid masking on M and N axis
    # m_offs_untagged and n_offs_untagged can contains addresses outside matrix boundaries
    # modulo operation is used to wrap around the indices that go beyond the matrix boundaries
    # The value loaded are not ok but at least we are not reading outside the A/B matrices
    # Then, during storing in C a mask is used and the results related to these wrong values is discarded!
    # Regarding max_contiguous and multiple_of, they are used to force the compiler to vectorize loads
    # multiple_of indicates that the first element of rm / rn is a multiple of BLOCK_M / BLOCK_N
    # max_contiguous indicates that the range is a block of BLOCK_M / BLOCK_N contiguous elements
    m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % M, BLOCK_M), BLOCK_M)
    n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % N, BLOCK_N), BLOCK_N)

    k_range_offs = tl.arange(0, BLOCK_K)

    A = A + (m_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    B = B + (k_range_offs[:, None] * b_k_stride + n_offs[None, :] * b_n_stride)
    scale = scale + n_offs[None, :]


    s = tl.load(scale)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k in range(K, 0, -BLOCK_K):
        if K_LOAD_MASK_NEEDED:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
            b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)

        b = (b.to(ACC_TYPE) / s).to(ACC_TYPE)

        acc += tl.dot(a, b).to(ACC_TYPE)

        A += BLOCK_K * a_k_stride
        B += BLOCK_K * b_k_stride

    # acc = acc / s
    
    if HAS_BIAS:
        bias = tl.load(bias + n_offs, mask=n_offs < N, other=0.0).to(ACC_TYPE)
        acc += bias[None, :]

    # optional: fused activation (while the data is in shared memory)
    if ACTIVATION == "gelu":
        acc = gelu(acc)

    # write back result
    C = C + m_offs[:, None] * output_m_stride + n_offs[None, :] * output_n_stride
    c_ptr_mask = (m_offs < M)[:, None] & (n_offs < N)[None, :]
    tl.store(C, acc, mask=c_ptr_mask)


def w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation="",
) -> torch.Tensor:
    """
    x: (M, K) bf16
    weight: (N, K) uint8
    scale: (N/32, 1) bf16
    bias: (N,) bf16
    """

    x_ = x if x.ndim == 2 else x.flatten(0, 1)

    if bias is not None:
        assert x.dtype == bias.dtype, f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"
    assert x_.shape[1] == weight.shape[1], f"Incompatible dimensions: {x_.shape} - {weight.shape}"

    assert bias is None or bias.is_contiguous()
    assert bias is None or bias.shape[0] == weight.shape[0], "Incompatible dimensions in between weight and bias"
    assert weight.is_contiguous()
    assert x.is_contiguous()

    M, K = x_.shape
    N, K = weight.shape

    outputs = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

    _w16a16_linear_kernel[grid](
        outputs,
        x_,
        weight,  # data ptrs
        scale.repeat_interleave(32, 1),
        bias if bias is not None else x,  # auto skip bias if not present
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        output_m_stride=outputs.stride(0),  # strides
        output_n_stride=outputs.stride(1),
        a_m_stride=x_.stride(0),
        a_k_stride=x_.stride(1),
        b_n_stride=weight.stride(0),
        b_k_stride=weight.stride(1),
        HAS_BIAS=bias is not None,  # optional fused bias
        ACTIVATION=activation if not None else x,  # optional fused activation
        GROUP_M=8,  # speed optimization: group the programs
        ACC_TYPE=tl.float16,
    )

    outputs = outputs if x.ndim == 2 else outputs.reshape(x.shape[0], -1, N)
    return outputs