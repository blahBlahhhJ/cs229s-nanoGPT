from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class OptimizationConfig:
    fuse_bias_dropout_residual: bool = False
    fuse_linear_bias_act: bool = False


def f(model, idx, pos, temperature=1.0, top_k=None):
    logits, _ = model.decode(idx, pos)
    logits = logits.squeeze(1) / temperature

    if top_k == 1:
        idx_next = logits.argmax(-1, keepdim=True)
    else:
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next

prefill = torch.compile(f, fullgraph=True, dynamic=True)
decode_one_token = torch.compile(f, mode='reduce-overhead', fullgraph=True)
