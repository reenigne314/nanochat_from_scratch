import torch
import torch.nn as nn

print("CUDA available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Testing basic GPU operations...")

# Test 1: Simple tensor operations
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = x @ y
print(f"Simple matmul OK: {z.shape}")

# Test 2: Embedding layer (part of your model)
emb = torch.nn.Embedding(10000, 256, device=device)
ids = torch.randint(0, 10000, (10, 32), device=device)
out = emb(ids)
print(f"Embedding layer OK: {out.shape}")

'''# Test 3: Attention block components
linear = torch.nn.Linear(256, 256, bias=False).to('cuda')
attn = linear(out)
print(f"Linear layer OK: {attn.shape}")'''

print("All basic GPU tests passed!")


'''
# Create a linear layer: 3 input features -> 2 output features
linear_layer = nn.Linear(3, 2, bias=True)

# Examine the parameters
print("Weight shape:", linear_layer.weight.shape)  # (2, 3)
print("Bias shape:", linear_layer.bias.shape)      # (2,)
# PyTorch automatically initialized weights!
print("Default weight matrix:")
print(linear_layer.weight)

# Create input: batch of 4 samples, each with 3 features
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]])  # Shape: (4, 3)

# Forward pass
y = linear_layer(x)  # Shape: (4, 2)

print("Output shape:", y.shape)  # (4, 2)
print("Output:", y)
# Manually compute one output:
# For first sample: y₁ = [1,2,3]·Wᵀ + b
# = [1*w₁₁ + 2*w₁₂ + 3*w₁₃ + b₁, 
#    1*w₂₁ + 2*w₂₂ + 3*w₂₃ + b₂]'''