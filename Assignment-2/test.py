import torch
import numpy as np

# Make a sample torch tensor
data = torch.randn(10, 10)
# Unsqeeze the tensor
samples = data.unsqueeze(1)

print(data.size())
print(samples.size())
# Print the first 5 elements of the tensor
print(data[0])
print(samples[0])