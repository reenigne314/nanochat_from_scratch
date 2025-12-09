import torch
import torch.nn as nn

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
#    1*w₂₁ + 2*w₂₂ + 3*w₂₃ + b₂]