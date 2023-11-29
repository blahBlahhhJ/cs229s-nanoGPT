import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from optimization.quantization import QuantizedLinear, QuantizeConfig
from optimization.fusion import FusedLinearBiasDropoutResidual

ref = torch.nn.Linear(4096, 1024).cuda()
fused = FusedLinearBiasDropoutResidual(4096, 1024).cuda()
fused.weight = ref.weight
fused.bias = ref.bias

x = torch.rand(1, 1, 4096).cuda()
r = torch.rand(1, 1, 1024).cuda()

print(ref(x) + r)
print(fused(x, r))

# fc = QuantizedLinear(256, 256, QuantizeConfig()).cuda()
# x = torch.rand(64, 256).cuda()
# fc.to(torch.float16)
# x = x.to(torch.float16)
# print('x', x[0, :10])
# print('fp16 w', fc.weight.data[0, :10])
# print('fp16 y', torch.round(fc(x), decimals=4)[0, :10])

# fc.quantize()
# print('int8', fc.quantizer.dequantize(fc.weight.data, fc.scale, fc.zero_point)[0, :10])
# # print('int8 s', fc.scale)
# print('int8 y', fc(x, triton=True)[0, :10])
# print('int8 yy', torch.round(torch.matmul(x, fc.quantizer.dequantize(fc.weight.data, fc.scale, fc.zero_point).T) + fc.bias, decimals=4)[0, :10])