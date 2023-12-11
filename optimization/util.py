from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class OptimizationConfig:
    fuse_bias_dropout_residual: bool = False
    fuse_linear_bias_act: bool = False
    inference_batch_size: int = 1
    inference_seq_len: int = 512


def prefill(model, idx, pos, temperature=1.0, top_k=None):
    logits = model.decode(idx, pos)
    logits = logits.squeeze(1) / temperature

    if top_k == 1:
        return logits
    else:
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        return probs

def decode_one_token(model, idx, pos, temperature=1.0, top_k=None):
    assert pos.shape[-1] == 1
    logits = model.decode(idx, pos)
    logits = logits.squeeze(1) / temperature

    if top_k == 1:
        return logits
    else:
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        return probs

# prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
# decode_one_token = torch.compile(decode_one_token, mode='reduce-overhead', fullgraph=True)
