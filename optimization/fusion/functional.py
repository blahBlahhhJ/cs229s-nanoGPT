import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization.fusion.triton import (
    bias_dropout_residual as fused_bias_dropout_residual,
)
