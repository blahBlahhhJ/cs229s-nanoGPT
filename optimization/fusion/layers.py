import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization.fusion.triton import (
    linear_bias_act,
    linear_bias_dropout_residual,
)


class FusedLinearBiasAct(nn.Linear):
    def __init__(self, in_features: int, out_features: int, fuse_gelu: bool = False, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

        self.fuse_gelu = fuse_gelu

    def forward(self, input):
        return linear_bias_act(
            input,
            self.weight,
            self.bias,
            activation=1 if self.fuse_gelu else 0,
            trainable_weight=self.weight.requires_grad,
            trainable_bias=self.bias is not None and self.bias.requires_grad,
            save_activation_inputs=self.training and input.requires_grad and self.fuse_gelu,
        )
    
class FusedLinearBiasDropoutResidual(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, input, residual):
        return linear_bias_dropout_residual(
            input,
            self.weight,
            residual,
            self.bias,
        )
