from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    fuse_bias_dropout_residual: bool = True
    fuse_linear_bias_act: bool = True