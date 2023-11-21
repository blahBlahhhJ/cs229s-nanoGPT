import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import (
    QuantizeConfig,
    Quantizer,
)
from optimization.quantization.triton import w8a16_linear


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, qconfig: QuantizeConfig, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

        self.qconfig = qconfig
        self.quantizer = Quantizer(qconfig)  # hardcode fp16
        self.quantized = False  # lazy quantization

        self.register_buffer('scale', None, persistent=False)
        self.register_buffer('zero_point', None, persistent=False)

    def quantize(self):
        self.weight.data, self.scale, self.zero_point = self.quantizer.quantize(self.weight.data)
        self.quantized = True

    def forward(self, input, triton=False):
        if self.quantized:
            if self.qconfig.sym and triton:
                weight = self.quantizer.cast_quantized_to_real_type(self.weight.data)
                res = w8a16_linear(input, weight, scale=self.scale, bias=self.bias, activation="")
            else:
                dequantized_weight = self.quantizer.dequantize(self.weight.data, self.scale, self.zero_point)
                res = F.linear(input, dequantized_weight, bias=self.bias)
            return res
        else:
            return super().forward(input)
        
        
