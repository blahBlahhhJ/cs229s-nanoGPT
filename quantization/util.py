import torch
from torch import nn, Tensor
from typing import Tuple
from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    sym: bool = True
    dtype: torch.dtype = torch.bfloat16
    num_bits: int = 8
    group_dim: int = 0
    group_size: int = 32


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
    

class Quantizer:
    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config
        assert self.config.num_bits == 4 or self.config.num_bits == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config.group_dim == 0, 'Only rowwise group split is supported.'

    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Arguments:
            tensor: (out_features, in_features)
        Returns:
            quantized_tensor: (out_features, in_features) uint8 -> bf16
            scale: (num_groups, 1) bf16
            min_value: (num_groups, 1) bf16
        """
        assert tensor.shape[self.config.group_dim] % self.config.group_size == 0 \
            , f'Tensor shape: {tensor.shape} quantization config {self.config}'

        shape = tensor.shape
        num_groups = shape[self.config.group_dim] // self.config.group_size
        tensor = tensor.view(num_groups, -1)

        quantized_tensor, scale, min_value = self._quantize_int8(tensor)
        quantized_tensor = quantized_tensor.view(shape)

        # .view(self.dtype) is a fake type cast to put into nn.Parameter (int not supported)
        if self.config.num_bits == 4:
            return self._compress_uint8_to_uint4(quantized_tensor).view(self.config.dtype), scale, min_value
        if self.config.num_bits == 8:
            return quantized_tensor.view(self.config.dtype), scale, min_value

        assert False, 'Unsupported quantization bits {}'.format(self.config.num_bits)

    def _quantize_int8(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.config.sym:
            q_range = 2 ** (self.config.num_bits - 1) - 1
            absmax_value = tensor.abs().amax(dim=self.config.group_dim + 1, keepdim=True)
            min_value = None

            scale = q_range / absmax_value
            
            tensor = tensor.mul_(scale)
            tensor = tensor_round(tensor_clamp(tensor, -q_range, q_range)).to(torch.int8)
        else:
            q_range = 2**self.config.num_bits - 1
            min_value = tensor.amin(dim=self.config.group_dim + 1, keepdim=True)
            max_value = tensor.amax(dim=self.config.group_dim + 1, keepdim=True)

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

    def dequantize(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        """
        Arguments:
            tensor: (out_features, in_features) uint8 -> bf16
            quant_scale: (num_groups, 1) bf16
            quant_min: (num_groups, 1) bf16
        Returns:
            dequantized_tensor: (out_features, in_features)
        """
        tensor = self.cast_quantized_to_real_type(tensor)

        if self.config.num_bits == 4:
            tensor = self._decompress_uint4_to_uint8(tensor.view(torch.uint8))
        else:
            assert self.config.num_bits == 8, 'Unsupported quantization bits {}'.format(self.config.num_bits)
            
        shape = tensor.shape
        num_groups = shape[self.config.group_dim] // self.config.group_size
        tensor = tensor.view(num_groups, -1)

        dequantized_tensor = self._dequantize_int8(tensor, quant_scale, quant_min).view(shape)
        return dequantized_tensor
    
    def cast_quantized_to_real_type(self, tensor: Tensor):
        if self.config.sym:
            tensor = tensor.view(torch.int8)
        else:
            tensor = tensor.view(torch.uint8)
        return tensor

    def _dequantize_int8(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        if self.config.sym:
            assert tensor.dtype == torch.int8
            data = torch.zeros_like(tensor, dtype=self.config.dtype, device=tensor.device)
            data = data.copy_(tensor)
            data = data.div_(quant_scale)
        else:
            assert tensor.dtype == torch.uint8
            data = torch.zeros_like(tensor, dtype=self.config.dtype, device=tensor.device)
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
