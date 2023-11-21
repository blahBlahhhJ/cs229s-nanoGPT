import torch
from torch import nn, Tensor
import torch.nn.functional as F
import time
from typing import Dict, Callable, Union, Tuple

import numpy as np


class Quantizer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert tensor.shape[self.config['group_dim']] % self.config['group_size'] == 0 \
            , f'Tensor shape: {tensor.shape} quantization config {self.config}'

        tensor = torch.clone(tensor)

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        quantized_tensor, scale, min_value = self._quantize_int8(tensor)
        quantized_tensor = quantized_tensor.view(shape)

        if self.config['num_bits'] == 4:
            return self._compress_uint8_to_uint4(quantized_tensor), scale, min_value
        if self.config['num_bits'] == 8:
            return quantized_tensor, scale, min_value

        assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

    def _quantize_int8(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q_range = 2**self.config['num_bits'] - 1
        min_value = tensor.amin(dim=self.config['group_dim'] + 1, keepdim=True)
        max_value = tensor.amax(dim=self.config['group_dim'] + 1, keepdim=True)

        scale = q_range / (max_value - min_value)

        tensor = tensor.sub_(min_value).mul_(scale)
        tensor = tensor_round(tensor_clamp(tensor, 0, q_range)).to(torch.uint8)
        return tensor, scale, min_value

    def _compress_uint8_to_uint4(self, tensor: Tensor) -> Tensor:
        assert tensor.shape[-1] % 2 == 0

        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] // 2

        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data = torch.bitwise_or(tensor[..., 0::2].bitwise_left_shift(4), tensor[..., 1::2])

        return data


class DeQuantizer:

    def __init__(self, config: Dict, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def dequantize(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        if self.config['num_bits'] == 4:
            tensor = self._decompress_uint4_to_uint8(tensor)
        elif self.config['num_bits'] != 8:
            assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        dequantized_tensor = self._dequantize_int8(tensor, quant_scale, quant_min).view(shape)
        return dequantized_tensor

    def _dequantize_int8(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        assert tensor.dtype == torch.uint8
        data = torch.zeros_like(tensor, dtype=self.dtype, device=tensor.device)
        data = data.copy_(tensor)
        data = data.div_(quant_scale).add_(quant_min)

        return data

    def _decompress_uint4_to_uint8(self, tensor: Tensor) -> Tensor:
        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] * 2
        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data[..., 0::2] = tensor.bitwise_right_shift(4)
        data[..., 1::2] = tensor.bitwise_and(0xF)

        return data
    
def concat_to_compat_param(quantized_weight: Tensor,
                           quant_scale: Tensor,
                           quant_min: Tensor,
                           return_param: bool = True) -> Union[nn.Parameter, Tensor]:
    shape_wieght = quantized_weight.shape
    shape_scale = quant_scale.shape
    shape_min = quant_min.shape

    quantized_weight = torch.flatten(quantized_weight)
    quant_scale = torch.flatten(quant_scale)
    quant_min = torch.flatten(quant_min)

    def deconcat_individual_tensors(shape_wieght: torch.Size, shape_scale: torch.Size,
                                    shape_min: torch.Size) -> Callable:

        def fn(compat_tensor: nn.Parameter) -> Tuple[Tensor, Tensor, Tensor]:
            weight = torch.narrow(compat_tensor, 0, 0, shape_wieght.numel()).view(shape_wieght)
            scale = torch.narrow(compat_tensor, 0, shape_wieght.numel(), shape_scale.numel()).view(shape_scale)
            min_val = torch.narrow(compat_tensor, 0,
                                   shape_wieght.numel() + shape_scale.numel(), shape_min.numel()).view(shape_min)

            return weight, scale, min_val

        return fn

    compat_tensor = torch.concat([quantized_weight, quant_scale, quant_min])
    if return_param:
        compat_tensor = nn.Parameter(compat_tensor, requires_grad=False)
    compat_tensor.deconcat = deconcat_individual_tensors(shape_wieght, shape_scale, shape_min)

    return compat_tensor


def _quantize_param(param: nn.Parameter, quant_config: Dict):
    assert not hasattr(param, 'weight_quantized'), 'Parameter has already been quantized.'
    quantizer = Quantizer(quant_config)
    dequantizer = DeQuantizer(quant_config, param.dtype)

    quantized_weight, quant_scale, quant_min = quantizer.quantize(param.data)

    quantized_weight = quantized_weight.view(param.dtype)
    quant_scale = quant_scale.view(param.dtype)
    quant_min = quant_min.view(param.dtype)

    quantized_compat_tensor = concat_to_compat_param(quantized_weight, quant_scale, quant_min)
    param.data = quantized_compat_tensor
    param.deconcat = quantized_compat_tensor.deconcat

    param.quantizer = quantizer
    param.dequantizer = dequantizer
    setattr(param, 'weight_quantized', True)

def tensor_clamp(tensor: Tensor, min, max) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 clamp
        return tensor.to(dtype=torch.float32).clamp_(min, max).to(dtype=torch.float16)
    else:
        return tensor.clamp_(min, max)


def tensor_round(tensor: Tensor) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 round
        return tensor.to(dtype=torch.float32).round_().to(dtype=torch.float16)
    else:
        return tensor.round_()

quantized_weight_registry = {}
# deal with weight sharing
def get_quantized_weight_wrapper(model, pre_quant_weight: nn.Parameter, quantize_weight_fn: Callable) -> nn.Parameter:
    if id(pre_quant_weight) in quantized_weight_registry:
        compat_tensor = quantized_weight_registry[id(pre_quant_weight)]

        return quantized_weight_registry[id(pre_quant_weight)]
    else:
        quantized_weights, quant_scale, quant_min = quantize_weight_fn()
        quantized_weight_registry[id(pre_quant_weight)] = concat_to_compat_param(quantized_weights, quant_scale,
                                                                                 quant_min)
        return quantized_weight_registry[id(pre_quant_weight)]


def get_quantize_weight_fn(quantizer: Quantizer, pre_quant_weight: nn.Parameter) -> Callable:

    def func() -> Tuple[nn.Parameter, Tensor, Tensor]:
        quantized_weights, quant_scale, quant_min = quantizer.quantize(pre_quant_weight.data)
        # A temporary hack as zero Zero3 assume all model weights has the same type. in all_gather_coalesced.get_only_unique_item
        quantized_weights = quantized_weights.view(pre_quant_weight.dtype)
        quant_scale = quant_scale.type(pre_quant_weight.dtype)
        quant_min = quant_min.type(pre_quant_weight.dtype)
        return quantized_weights, quant_scale, quant_min

    return func




config = {'num_bits': 8, 'symmetric': False, 'group_dim': 1, 'group_size': 64}
w = nn.Parameter(torch.rand(1024, 4096, dtype=torch.float16))
quant = Quantizer(config)

quant_fn = get_quantize_weight_fn(quant, w)
quantized_weights, quant_scale, quant_min = quant_fn()
import IPython; IPython.embed()






# for i in range(1000):
#     w = torch.rand(1024, 4096, dtype=torch.float16).cuda()
#     x = torch.rand(1, 1024, dtype=torch.float16).cuda()

#     y = torch.matmul(x, w)

# fpt = []
# for i in range(100):
#     w = torch.rand(1024, 4096, dtype=torch.float16).cuda()
#     x = torch.rand(10, 1024, dtype=torch.float16).cuda()

#     t = time.time()
#     y = torch.matmul(x, w)
#     fpt.append(time.time() - t)

# intt = []
# s = torch.tensor([1./127], dtype=torch.float16)
# for i in range(100):
#     w = torch.randint(0, 256, (1024, 4096), dtype=torch.uint8).cuda()
#     x = torch.rand(10, 1024, dtype=torch.float16).cuda()

#     t = time.time()

#     y = torch.matmul(x, w)
#     intt.append(time.time() - t)

# print(np.mean(fpt))
# print(np.mean(intt))