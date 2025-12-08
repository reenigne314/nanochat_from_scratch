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

