import torch

# Suppose we have tiny example:
B, T, C = 1, 2, 12  # 12 = 3 heads × 4 dim/head
n_head, head_dim = 3, 4

# Simulate linear layer output (flattened heads)
# For token 1: [h1_d1, h1_d2, h1_d3, h1_d4, h2_d1, ..., h3_d4]
# For token 2: same pattern
q_proj = torch.tensor([
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],    # Token 1
     [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]  # Token 2
])
# Shape: (1, 2, 12)

print("Original q_proj:")
print(q_proj)
# Reshape to separate heads
q = q_proj.view(B, T, n_head, head_dim)
print(q.shape)  # (1, 2, 3, 4)
print(q[0, 0, :, :])