import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
import math

import torch
import torch.nn as nn

from nanochat.gpt import GPTConfig, GPT

def test_gpt_module():
    print("=" * 80)
    print("TESTING GPT MODULE")
    print("=" * 80)
    
    # Test 1: Create config and model
    print("\n1. Creating model with config...")
    config = GPTConfig()
    
    model = GPT(config)
    '''model.init_weights()
    model = model.to('cuda')'''
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
    print(f"✓ Rotary embeddings dtype: {model.cos.dtype}")'''
    
    # Test 4: Initialize weights
    print("\n4. Initializing weights...")
    model.init_weights()
    
    # Check that specific weights are zero
    lm_head_mean = model.lm_head.weight.mean().item()
    assert abs(lm_head_mean) < 1e-6, f"lm_head weights not zero: mean={lm_head_mean}"
    print(f"✓ lm_head weights zeroed: mean={lm_head_mean:.6f}")
    
    # Check block output projections are zero
    for i, block in enumerate(model.transformer.h):
        attn_proj_mean = block.attn.c_proj.weight.mean().item()
        mlp_proj_mean = block.mlp.c_proj.weight.mean().item()
        assert abs(attn_proj_mean) < 1e-6, f"Block {i} attn.c_proj not zero: {attn_proj_mean}"
        assert abs(mlp_proj_mean) < 1e-6, f"Block {i} mlp.c_proj not zero: {mlp_proj_mean}"
    print(f"✓ All c_proj weights zeroed")    

    # Test 5: Check weight initialization statistics
    print("\n5. Checking weight initialization statistics...")
    
    # Collect all linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    print(f"  Found {len(linear_layers)} linear layers")
    
    # Check each linear layer's initialization
    for name, layer in linear_layers:
        weight = layer.weight
        mean = weight.mean().item()
        std = weight.std().item()
        
        # Calculate expected std based on fan_in/fan_out
        fan_in = weight.size(1)
        fan_out = weight.size(0)
        expected_std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        
        # Skip zero-initialized layers
        if "c_proj" in name or "lm_head" in name:
            if abs(mean) > 1e-6 or abs(std) > 1e-6:
                print(f"  ⚠️ {name}: Should be zero, but mean={mean:.6f}, std={std:.6f}")
            continue
            
        # Check mean is near 0
        if abs(mean) > 0.1:
            print(f"  ⚠️ {name}: Mean too high: {mean:.6f}")
        
        # Check std is reasonable
        if abs(std - expected_std) > 0.01:
            print(f"  ⚠️ {name}: Std deviates: {std:.6f} vs expected {expected_std:.6f}")
    
    print(f"✓ Weight initialization check completed")

    # Test 6: Check embedding initialization
    print("\n6. Checking embedding initialization...")
    embedding = model.transformer.wte.weight
    emb_mean = embedding.mean().item()
    emb_std = embedding.std().item()
    
    # Embeddings should have std ~1.0 (from _init_weights)
    if abs(emb_std - 1.0) > 0.1:
        print(f"  ⚠️ Embeddings: Std {emb_std:.6f} not close to 1.0")
    else:
        print(f"✓ Embeddings initialized with std ~1.0: {emb_std:.6f}")

    # Test 7: Test forward pass
    print("\n7. Testing forward pass...")
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, \
        f"Output shape mismatch: {logits.shape} != {expected_shape}"
    print(f"✓ Forward pass output shape: {logits.shape}")

    # Test 8: Check device handling
    print("\n8. Checking device handling...")
    device = model.get_device()
    print(f"  Model device: {device}")
    
    # Test moving to CPU (if not already)
    model.cpu()
    cpu_device = model.get_device()
    assert cpu_device.type == 'cpu', f"Model not on CPU: {cpu_device}"
    print(f"✓ Model can be moved to CPU")

    # Test 9: Check BF16 conversion (on CUDA if available)
    print("\n9. Checking BF16 conversion...")
    if torch.cuda.is_available():
        model.cuda()
        # Re-initialize weights to trigger BF16 conversion
        model.init_weights()
        
        # Check wte dtype on CUDA
        if model.transformer.wte.weight.device.type == "cuda":
            wte_dtype = model.transformer.wte.weight.dtype
            if wte_dtype == torch.bfloat16:
                print(f"✓ wte converted to bfloat16 on CUDA")
            else:
                print(f"  ⚠️ wte dtype on CUDA: {wte_dtype} (expected bfloat16)")
    else:
        print("  Skipping CUDA/BF16 test (CUDA not available)")

    # Test 10: Parameter count
    print("\n10. Calculating parameter count...")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    total_params = count_parameters(model)
    
    # Calculate expected parameters
    # 1. wte: vocab_size * n_embd
    wte_params = config.vocab_size * config.n_embd
    
    # 2. Each block:
    #    - attn.c_q: n_embd * (n_head * head_dim) = n_embd * n_embd = n_embd²
    #    - attn.c_k: n_embd * (n_kv_head * head_dim)
    #    - attn.c_v: same as c_k
    #    - attn.c_proj: n_embd * n_embd
    #    - mlp.c_fc: n_embd * (4 * n_embd)
    #    - mlp.c_proj: (4 * n_embd) * n_embd
    head_dim = config.n_embd // config.n_heads
    
    attn_q_params = config.n_embd * (config.n_heads * head_dim)  # = n_embd²
    attn_kv_params = config.n_embd * (config.n_kv_head * head_dim)
    attn_proj_params = config.n_embd * config.n_embd
    mlp_fc_params = config.n_embd * (4 * config.n_embd)
    mlp_proj_params = (4 * config.n_embd) * config.n_embd
    
    block_params = attn_q_params + attn_kv_params * 2 + attn_proj_params + mlp_fc_params + mlp_proj_params
    total_blocks_params = block_params * config.n_layers
    
    # 3. lm_head: n_embd * vocab_size
    lm_head_params = config.n_embd * config.vocab_size
    
    expected_params = wte_params + total_blocks_params + lm_head_params
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected parameters: {expected_params:,}")
    
    if total_params == expected_params:
        print(f"✓ Parameter count matches expectation")
    else:
        print(f"  ⚠️ Parameter count mismatch!")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_gpt_module()