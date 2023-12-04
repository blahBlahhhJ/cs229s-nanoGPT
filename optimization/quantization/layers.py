import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import (
    QuantizeConfig,
    Quantizer,
)
from optimization.quantization.triton import w8a16_linear


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, qconfig: QuantizeConfig):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.qconfig = qconfig
        self.quantizer = Quantizer(qconfig)  # default to fp16

        self.register_buffer('weight', torch.empty(
            (out_features, in_features), 
            dtype=self.quantizer.quantized_dtype
        ))
        self.register_buffer('scale', torch.ones(
            (out_features * in_features // self.qconfig.group_size, 1),
            dtype=self.qconfig.dtype
        ))
        self.register_buffer('zero_point', None)
        self.register_buffer('bias', None)

    def quantize(self, weight, bias=None):
        self.weight, self.scale, self.zero_point = self.quantizer.quantize(weight)
        if bias is not None:
            self.bias = bias

    def forward(self, input):
        dequantized_weight = self.quantizer.dequantize(self.weight, self.scale, self.zero_point)
        res = F.linear(input, dequantized_weight, bias=self.bias)
        return res
        
        
