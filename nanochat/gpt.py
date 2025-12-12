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

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
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
        assert self.n_kv_head <= self.n_heads and self.n_heads % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_heads * self.head_dim, bias=False)
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
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layers)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # to support meta device initialization
        # Black box yet to understand why
        self.rotary_seq_len = config.sequence_length * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out lm_head weights as in GPT-2
        torch.nn.init.zeros_(self.lm_head.weight)
        # zer out c_proj weights in each attention layer
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init rotary embeddings again to make sure
        head_dim = self.config.n_embd // self.config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # cast the embeddings to the same dtype as bf16 as it saves memory
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.weight.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect device from model parameters
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # calculate the position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        #calculate the position angles
        freqs = torch.outer(t, inv_freq)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        return cos, sin
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        # based on chichilla scaling law paper: https://arxiv.org/abs/2204.02311
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layers, self.config.n_heads, self.config.n_embd // self.config.n_heads, self.config.sequence_length
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # seperate parameters into 3 different parts
        matrix_params = list(self.transformer.h.parameters())  # all transformer block params
        embedding_params = [self.transformer.wte.weight]  # token embedding
        lm_head_params = [self.lm_head.weight]  # lm head
        # setup optimizers with different learning rates
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling learning rates by 1/sqrt({model_dim}) = {dmodel_lr_scale:.4f}")
            adam_groups = [
                dict(params=lm_head_params, lr=embedding_lr * dmodel_lr_scale),
                dict(params=embedding_params, lr=unembedding_lr * dmodel_lr_scale),
            ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers        

    def forward(self, idx, targets = None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        #assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"        
        # if kv cache exists 
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # (1, T, 1, head_dim/2)

        # Token embeddings
        x = self.transformer.wte(idx)  # (B, T, C)
        x = norm(x)  # norm after token embedding
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float() # use tf32/fp32 for logits
        logits = softcap * torch.tanh(logits / softcap) # logits softcap

        # Forward the lm_head (compute logits)
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            return logits