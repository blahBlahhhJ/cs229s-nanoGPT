import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from optimization.quantization import QuantizedLinear, QuantizeConfig

fc = torch.nn.Linear(256, 256).cuda()
qfc = QuantizedLinear(256, 256, QuantizeConfig()).cuda()
x = torch.rand(64, 256).cuda()
fc.to(torch.bfloat16)
x = x.to(torch.bfloat16)

print('x', x[0, :10])
print('fp16 w', fc.weight.data[0, :10])
print('fp16 y', torch.round(fc(x), decimals=4)[0, :10])

qfc.quantize(fc.weight.data, fc.bias.data)
# print('int8', qfc.quantizer.dequantize(qfc.weight.data, qfc.scale, qfc.zero_point)[0, :10])
# print('int8 s', fc.scale)
print('int8 y', qfc(x)[0, :10])
print('int8 yy', torch.round(torch.matmul(x, qfc.quantizer.dequantize(qfc.weight.data, qfc.scale, qfc.zero_point).T) + qfc.bias/qfc.scale, decimals=4)[0, :10])