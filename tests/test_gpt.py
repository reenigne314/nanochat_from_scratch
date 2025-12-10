import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import torch

from nanochat.gpt import GPTConfig, GPT

def test_gpt_module():
    print("=" * 80)
    print("TESTING GPT MODULE")
    print("=" * 80)
    
    # Test 1: Create config and model
    print("\n1. Creating model with config...")
    config = GPTConfig()
    
    model = GPT(config)
    model.init_weights()
    model = model.to('cuda')
    print(f"✓ Model created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 2: Check model structure
    print("\n2. Checking model structure...")
    
    # Check transformer components
    assert hasattr(model.transformer, 'wte'), "Missing wte (word token embedding)"
    assert hasattr(model.transformer, 'h'), "Missing h (blocks)"
    assert len(model.transformer.h) == config.n_layers, f"Expected {config.n_layers} blocks, got {len(model.transformer.h)}"
    print(f"✓ Transformer structure: OK")    

    # Check wte dimensions
    wte_weight = model.transformer.wte.weight
    assert wte_weight.shape == (config.vocab_size, config.n_embd), \
        f"wte shape mismatch: {wte_weight.shape} != ({config.vocab_size}, {config.n_embd})"
    print(f"✓ wte shape: {wte_weight.shape}")
    
    # Check lm_head dimensions
    lm_head_weight = model.lm_head.weight
    assert lm_head_weight.shape == (config.vocab_size, config.n_embd), \
        f"lm_head shape mismatch: {lm_head_weight.shape} != ({config.vocab_size}, {config.n_embd})"
    print(f"✓ lm_head shape: {lm_head_weight.shape}")
    
    # Test 3: Check rotary embeddings
    print("\n3. Checking rotary embeddings...")
    assert hasattr(model, 'cos'), "Missing cos buffer"
    assert hasattr(model, 'sin'), "Missing sin buffer"

    # Check rotary embedding dimensions
    expected_seq_len = config.sequence_length * 10
    head_dim = config.n_embd // config.n_heads
    assert model.cos.shape == (1, expected_seq_len, 1, head_dim / 2), \
        f"cos shape mismatch: {model.cos.shape} != (1, {expected_seq_len}, 1, {head_dim})"
    assert model.sin.shape == (1, expected_seq_len, 1, head_dim / 2), \
        f"sin shape mismatch: {model.sin.shape} != (1, {expected_seq_len}, 1, {head_dim})"
    print(f"✓ Rotary embeddings shape: {model.cos.shape}")

    print(model.transformer.wte.weight.device.type)
    '''
    # Check dtype (should be bfloat16)
    assert model.cos.dtype == torch.bfloat16, f"cos dtype is {model.cos.dtype}, expected bfloat16"
    assert model.sin.dtype == torch.bfloat16, f"sin dtype is {model.sin.dtype}, expected bfloat16"
    print(f"✓ Rotary embeddings dtype: {model.cos.dtype}")
    '''
if __name__ == "__main__":
    test_gpt_module()