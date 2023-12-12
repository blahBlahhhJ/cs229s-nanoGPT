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

def decode_n_tokens(model, idx, pos, num_new_tokens, device, temperature=1.0, top_k=None):
    idx_next = idx
    pos = torch.tensor(pos, dtype=torch.long, device=device)
    new_tokens = torch.empty(num_new_tokens, dtype=torch.long, device=device)
    new_probs = torch.empty(num_new_tokens, model.config.vocab_size, device=device)

    for i in range(num_new_tokens):
        probs = decode_one_token(
            model, idx_next, pos, temperature, top_k
        )
        if top_k == 1:
            idx_next = probs.argmax(-1, keepdim=True)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        
        pos += 1
        new_tokens[i] = idx_next
        new_probs[i] = probs
    return new_tokens, new_probs

# prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
# decode_one_token = torch.compile(decode_one_token, mode='reduce-overhead', fullgraph=True)
