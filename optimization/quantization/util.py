import torch
from torch import nn, Tensor
from typing import Tuple
from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    quantize_from_init: bool = False
    sym: bool = True
    dtype: torch.dtype = torch.float16
    num_bits: int = 8
    group_size: int = 32    # not used for now
    

class Quantizer:
    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config
        assert self.config.num_bits == 8, 'Only INT8 quantization is supported.'

    @property
    def quantized_dtype(self):
        if self.config.sym:
            return torch.int8
        else:
            return torch.uint8

    @torch.no_grad
    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Arguments:
            tensor: (out_features, in_features)
        Returns:
            quantized_tensor: (out_features, in_features) int8
            scale: (out_features, 1) bf16
            min_value: (out_features, 1) bf16
        """
        quantized_tensor, scale, min_value = self._quantize_int8(tensor)

        return quantized_tensor, scale, min_value

    def _quantize_int8(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.config.sym:
            q_range = 2 ** (self.config.num_bits - 1) - 1
            absmax_value = tensor.abs().amax(dim=-1, keepdim=True)
            zero_point = None

            scale = q_range / absmax_value
            
            tensor = (
                tensor
                .mul_(scale)
                .round()
                .clamp_(-q_range, q_range)
                .to(torch.int8)
            )
        else:
            q_range = 2**self.config.num_bits - 1
            min_value = tensor.amin(dim=self.config.group_dim + 1, keepdim=True)
            max_value = tensor.amax(dim=self.config.group_dim + 1, keepdim=True)

            scale = q_range / (max_value - min_value)
            zero_point = min_value

            tensor = (
                tensor
                .sub_(min_value)
                .mul_(scale)
                .round()
                .clamp_(0, q_range)
                .to(torch.uint8)
            )

        return tensor, scale, zero_point

    @torch.no_grad
    def dequantize(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        """
        Arguments:
            tensor: (out_features, in_features) int8
            quant_scale: (out_features, 1) bf16
            quant_min: (out_features, 1) bf16
        Returns:
            dequantized_tensor: (out_features, in_features)
        """
        dequantized_tensor = self._dequantize_int8(tensor, quant_scale, quant_min)

        return dequantized_tensor

    def _dequantize_int8(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        assert tensor.dtype == self.quantized_dtype

        if self.config.sym:
            data = tensor.to(self.config.dtype).div_(quant_scale)
        else:
            data = tensor.to(self.config.dtype).div_(quant_scale).add_(quant_min)

        return data
