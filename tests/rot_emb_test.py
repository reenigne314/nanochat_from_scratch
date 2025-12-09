import torch
import math

# Original vectors
q = torch.tensor([1.0, 0.0])
k = torch.tensor([0.0, 1.0])

# Rotation angles (m=1, n=2)
m, n = 1, 2
theta = 1.0  # For simplicity

# Rotation matrices
def R(theta):
    return torch.tensor([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
    ])

R_m = R(m * theta)  # R(1)
R_n = R(n * theta)  # R(2)

print(f"Rotation matrix R(1):\n{R_m}")
print(f"Rotation matrix R(2):\n{R_n}")

# Method 1: Rotate then dot
q_rot = R_m @ q  # Matrix-vector multiply
k_rot = R_n @ k

print(f"Rotated q: {q_rot}")
print(f"Rotated k: {k_rot}")
dot1 = torch.dot(q_rot, k_rot)

# Method 2: Using the derived formula
R_diff = R((n - m) * theta)  # R(1)
k_transformed = R_diff @ k
dot2 = torch.dot(q, k_transformed)

print(f"Method 1 (rotate then dot): {dot1:.4f}")  # 0.6536
print(f"Method 2 (q·R(1)k): {dot2:.4f}")         # 0.6536