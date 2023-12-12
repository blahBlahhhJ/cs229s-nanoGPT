import torch
from torch import nn
import torch.nn.functional as F

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name_child, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name_child if name != '' else name_child
        ))
    return res

class PrunedLinear(nn.Linear):
    def __init__(self, in_features, *args, **kwargs):
        super().__init__(in_features=in_features, *args, **kwargs)

        self.register_buffer("prune_mask", torch.ones(in_features, dtype=torch.bool), persistent=False)

    def forward(self, input):
        return F.linear(input[:, self.prune_mask], self.weight[:, self.prune_mask], self.bias)
    
def replace_linear_with_pruned_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlinear = PrunedLinear(child.in_features, child.out_features, child.bias is not None)
            setattr(module, name, qlinear)
        else:
            replace_linear_with_pruned_linear(child)


class L2PruningHandler:
    def __init__(self, model):
        self.model = model

    def handle(self):
        replace_linear_with_pruned_linear(self.model)
