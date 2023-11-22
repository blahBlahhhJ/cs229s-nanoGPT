from torch import nn

def find_layers(self, module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name_child, child in module.named_children():
        res.update(self._find_layers(
            child, layers=layers, name=name + '.' + name_child if name != '' else name_child
        ))
    return res