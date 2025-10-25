import torch
import torch.nn.functional as F

# Test the failing case
input_torch = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 6)
weight_torch = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)

input_torch.requires_grad_(True)
weight_torch.requires_grad_(True)

# Try different padding strategies
print("=== No padding ===")
out_torch = F.conv1d(input_torch, weight_torch, padding=0)
print("Output shape:", out_torch.shape)
print("Output:", out_torch)

print("\n=== Same padding ===")
out_torch2 = F.conv1d(input_torch, weight_torch, padding=3)  # (kernel_size-1)
print("Output shape:", out_torch2.shape)
print("Output:", out_torch2)

# Test gradients with same padding
out_torch2.sum().backward()
print("Input grad with same padding:", input_torch.grad)
print("Expected grad at (0,0,3):", input_torch.grad[0,0,3])