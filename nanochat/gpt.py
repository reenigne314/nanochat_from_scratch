"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    sequence_length: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 6
    n_kv_head: int = 6  # for GQA
    n_embd: int = 768

def norm(x):
    # RMSNorm without learnable parameters
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (b, h, s, d) for multi-head attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split last dim into two halves
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    out = torch.cat([y1, y2], dim=3) # re-assemble
    out = out.to(x.dtype)
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        assert self.n_embd % self.n_heads == 0
        assert self.n_kv_head <= self.n_heads and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)

        # Apply kv cache for inference
        if kv_cache is not None:
            k, v = kv_cache.insert.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # num of query tokens in this forward pass
        Tk = k.size(2) # num of key/value tokens in cache

        # Attenntion: queries attend to keys/values auto-regressively
        enable_gqa = self.n_heads != self.n_kv_head # GQA: duplicate keys/values across heads
        if kv_cache is None or Tq == Tk:
            # During training or no new tokens to append to cache
            # And even if there is KV cache, but no new tokens to append
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # Inference with KV cache, one token at a time
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chuck of queries in this forward pass
            # First, each query attends to all cached keys/values (i.e., full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y            