# RMSNorm-Triton-Kernel

A Triton-accelerated implementation of RMS-Normalization for PyTorch.

## Overview

This repository provides a custom PyTorch module that implements RMS Normalization using Triton kernels. RMS Normalization normalizes input features by computing their root mean square (RMS) and scaling them with learned parameters. By leveraging Triton, the kernels are optimized for efficient execution on NVIDIA GPUs.

The repository includes:
- **Triton Forward Kernel:** Implements the forward pass, computing the RMS normalization for each row of the input.
- **Triton Backward Kernel:** Implements the backward pass, computing gradients with respect to both the input and the weight.
- **Custom Autograd Function:** Wraps the Triton kernels into a PyTorch-friendly API, enabling seamless integration with PyTorch's automatic differentiation.

## Features

- **High Performance:** Uses Triton to accelerate both forward and backward computations on the GPU.
- **Seamless Integration:** Easily plug into your PyTorch models as a drop-in replacement for standard normalization layers.
- **Customizable:** Supports configurable block sizes and epsilon for numerical stability.

## Requirements

Before using the RS Norm Triton Kernel, ensure you have the required dependencies installed:

- [PyTorch](https://pytorch.org/)
- [Triton](https://github.com/openai/triton)

```
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchvision==0.16.0+cu121
triton==2.1.0
```


## Basic Example
```
import torch
from RMSNormTritonKernel import RMSNormTritonKernel

# Input dimensions (Batch, Dim, Hidden)
B, D, H = 2, 3, 4
x = torch.randn(B, D, H, device='cuda', requires_grad=True)
grad_y = torch.randn(B, D, H, device='cuda')
weight = torch.randn(H, device='cuda', requires_grad=True)

eps = 1e-5

rmsnorm = RMSNormTritonKernel.apply
# Forward pass
y = rmsnorm(x, weight)

y.backward(grad_y)

print("Output:", y)
print("Gradients (x):", x.grad)
print("Gradients (weight):", weight.grad)
```
