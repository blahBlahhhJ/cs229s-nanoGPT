import torch.nn as nn

from .layers import QuantizedLinear


def replace_linear_weight_only_int8_per_channel(module, qconfig):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlinear = QuantizedLinear(child.in_features, child.out_features, qconfig)
            qlinear.quantize(child.weight.data, child.bias.data)
            setattr(module, name, qlinear)
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnly8BitHandler:
    def __init__(self, model):
        self.model = model

    def handle(self):
        replace_linear_weight_only_int8_per_channel(self.model, self.model.qconfig)
