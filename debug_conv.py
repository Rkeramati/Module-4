import minitorch
import torch
import torch.nn.functional as F

# Test the failing case from the test
input_tensor = minitorch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 6)
weight_tensor = minitorch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)

input_tensor.requires_grad_(True)
weight_tensor.requires_grad_(True)

print("Input:", input_tensor)
print("Weight:", weight_tensor)

# Forward pass
out = minitorch.Conv1dFun.apply(input_tensor, weight_tensor)
print("Output:", out)

# Backward pass
out.sum().backward()

print("Input grad:", input_tensor.grad)
print("Weight grad:", weight_tensor.grad)

# Compare with PyTorch
input_torch = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 6)
weight_torch = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)

input_torch.requires_grad_(True)
weight_torch.requires_grad_(True)

out_torch = F.conv1d(input_torch, weight_torch, padding=0)
print("\nPyTorch Output:", out_torch)

out_torch.sum().backward()
print("PyTorch Input grad:", input_torch.grad)
print("PyTorch Weight grad:", weight_torch.grad)