import torch
from torch import nn
import torch.nn.functional as F

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.
    """
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name_child, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name_child if name != '' else name_child
        ))
    return res

class PrunedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias, **kwargs):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)

        # self.register_buffer("active_indices", torch.arange(in_features), persistent=False)
        self.active_indices = torch.arange(in_features)

    def forward(self, input):
        # print("shapes", input.shape, self.weight.shape, self.prune_mask.shape)
        # mask applies to the last dimension of input
        # pruned numbers will have mask value of 1
        if self.training: # or self.active_indices.numel() == self.in_features:
            return super().forward(input)
        else:
            # hack: we only prune inputs during validation
            return F.linear(input[..., self.active_indices], self.pruned_weight, self.bias)
    
def replace_linear_with_pruned_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlinear = PrunedLinear(child.in_features, child.out_features, child.bias is not None)
            # copy the weight!
            qlinear.weight.data.copy_(child.weight.data)
            # copy the bias!
            if child.bias is not None:
                qlinear.bias.data.copy_(child.bias.data)
            setattr(module, name, qlinear)
        else:
            replace_linear_with_pruned_linear(child)


class L2PruningHandler:
    def __init__(self, model):
        self.model = model

    def handle(self):
        replace_linear_with_pruned_linear(self.model)
