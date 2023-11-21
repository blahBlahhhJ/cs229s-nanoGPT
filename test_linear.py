import torch

from quantization import QuantizedLinear, QuantizeConfig

fc = QuantizedLinear(256, 256, QuantizeConfig()).cuda()
x = torch.rand(64, 256).cuda()
fc.to(torch.bfloat16)
x = x.to(torch.bfloat16)
print('x', x[0, :10])
print('fp16 w', fc.weight.data[0, :10])
print('fp16 y', torch.round(fc(x), decimals=4)[0, :10])

fc.quantize()
print('int8', fc.quantizer.dequantize(fc.weight.data, fc.scale, fc.zero_point)[0, :10])
# print('int8 s', fc.scale)
print('int8 y', fc(x, triton=True)[0, :10])
print('int8 yy', torch.round(torch.matmul(x, fc.quantizer.dequantize(fc.weight.data, fc.scale, fc.zero_point).T) + fc.bias, decimals=4)[0, :10])