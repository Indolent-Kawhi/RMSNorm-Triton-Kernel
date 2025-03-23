import torch
import torch.nn as nn
import triton
import triton.language as tl

# Forward pass Triton kernel for RMSNorm.
@triton.jit
def RMSNorm_Triton_fwd(
        x_ptr: tl.pointer_type,         # Pointer to the input tensor.
        weight_ptr: tl.pointer_type,    # Pointer to the weight (scaling) vector.
        x_B_stride: tl.uint32,          # Stride for the batch dimension.
        x_D_stride: tl.uint32,          # Stride for the feature (row) dimension.
        ouput_ptr: tl.pointer_type,     # Pointer to the output tensor.
        ep: tl.float32,                 # Epsilon value for numerical stability.
        H: tl.uint32,                   # Number of elements in each row (feature dimension size).
        BLOCK_SIZE: tl.constexpr,       # Block size for parallel processing (usually next power of 2 of H).
):
    # Identify the block (batch and row index) for this kernel invocation.
    B_row_id = tl.program_id(0)       # Batch index from grid dimension 0.
    D_row_id = tl.program_id(1)       # Row/feature index from grid dimension 1
    
    # Create a range of offsets to index elements in the current block.
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute starting pointer offset for the input row based on batch and feature strides.
    strides = B_row_id * x_B_stride + D_row_id * x_D_stride
    x_start_ptr = x_ptr + strides
    # Compute pointers for each element in the block.
    x_ptrs = x_start_ptr + offsets

    # Compute corresponding starting pointer for the output row.
    out_start_ptr = ouput_ptr + strides
    out_ptrs = out_start_ptr + offsets

    # Compute pointer offsets for the weight vector.
    weight_ptrs = weight_ptr + offsets

    # Create a mask to handle cases where BLOCK_SIZE > H.
    mask = offsets < H
    # Load the row elements from the input tensor, using the mask.
    row = tl.load(x_ptrs, mask=mask, other=0)

    # Compute the sum of squares for RMS normalization.
    sum_sq = tl.sum(row * row)

    # Compute the mean of squares.
    mean_sq = sum_sq / H

    # Compute the normalization denominator: 1/sqrt(mean_sq + ep)
    denom = 1 / tl.sqrt(mean_sq + ep)
    
    # Load the weight values.
    weights = tl.load(weight_ptrs, mask=mask, other=0)

    # Normalize the row by multiplying with weights and the computed denominator.
    out = row * weights * denom

    # Store the result in the output tensor.
    tl.store(out_ptrs, out, mask=mask)


# Backward pass Triton kernel for RMSNorm.
@triton.jit
def RMSNorm_Triton_bwd(
        grad_x_ptr: tl.pointer_type,    # Pointer to the gradient tensor with respect to input (dx).
        grad_y_ptr: tl.pointer_type,    # Pointer to the gradient tensor from the next layer (dy).
        grad_w_ptr: tl.pointer_type,    # Pointer to the gradient tensor with respect to weight (dw).
        x_ptr: tl.pointer_type,         # Pointer to the original input tensor.
        weight_ptr: tl.pointer_type,    # Pointer to the weight (scaling) vector.
        x_B_stride: tl.uint32,          # Stride for the batch dimension.
        x_D_stride: tl.uint32,          # Stride for the feature (row) dimension.
        ep: tl.float32,                 # Epsilon value for numerical stability.
        H: tl.uint32,                   # Number of elements in each row (feature dimension size).
        BLOCK_SIZE: tl.constexpr,       # Block size for parallel processing.
):
    # Identify the current block's batch and row indices.
    B_row_id = tl.program_id(0)       # Batch index.
    D_row_id = tl.program_id(1)       # Row/feature index.
    
    # Create an array of offsets for block elements.
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < H               # Mask to ensure we only work on valid indices.

    # Compute the starting pointer offset for the current row in the input.
    strides = B_row_id * x_B_stride + D_row_id * x_D_stride
    x_start_ptr = x_ptr + strides
    x_ptrs = x_start_ptr + offset

    # Compute corresponding pointers for the weight vector.
    weight_ptrs = weight_ptr + offset

    # Compute pointers for the gradient from the next layer (dy).
    grad_y_start_ptr = grad_y_ptr + strides
    grad_y_ptrs = grad_y_start_ptr + offset

    # Compute pointers for the gradient with respect to input (dx).
    grad_x_start_ptr = grad_x_ptr + strides
    grad_x_ptrs = grad_x_start_ptr + offset

    # Compute pointers for the gradient with respect to weight (dw).
    grad_w_ptrs = grad_w_ptr + offset

    # Load the required vectors: input x, weight, and grad_y.
    x = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    grad_y = tl.load(grad_y_ptrs, mask=mask, other=0)

    # Compute the sum of squares of the input vector.
    sum_sq = tl.sum(x * x)

    # Compute the mean square value.
    mean_sq = sum_sq / H

    # Calculate the RMS normalization factor.
    rms = 1 / tl.sqrt(mean_sq + ep)

    # Compute intermediate gradient for weight (grad_g).
    grad_g = grad_y * x * rms

    # Atomically accumulate the gradient for weight.
    tl.atomic_add(grad_w_ptrs, grad_g, mask=mask)

    # Compute an intermediate term for input gradient calculation.
    xgdy = x * weight * grad_y
    xgdy_sum = tl.sum(xgdy) / H

    # Compute the gradient with respect to the input tensor.
    grad_x = grad_y * weight * rms - x * xgdy_sum * tl.math.fast_powf(rms, 3)
    # Store the computed gradient for input.
    tl.store(grad_x_ptrs, grad_x, mask=mask)


# Custom autograd function for RMSNorm using Triton kernels.
class RMSNormTritonKernel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, ep=1e-5):
        # x: Input tensor of shape (B, D, H)
        # weight: Scaling vector applied per element
        # ep: Epsilon for numerical stability
        
        B, D, H = x.shape
        
        # Prepare an empty output tensor.
        y = torch.empty_like(x, device=x.device)

        # Determine block size as the next power of 2 of H.
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        # Save tensors for backward computation.
        ctx.save_for_backward(x, weight)
        # Save epsilon value.
        ctx.ep = ep

        # Define the grid dimensions for Triton kernel: one block per (B, D) pair.
        grid = (B, D)
        RMSNorm_Triton_fwd[grid](
            x, weight, x.stride(0), x.stride(1), y, ep, H, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        
        return y

    @staticmethod
    def backward(ctx, dy):
        # Retrieve saved tensors.
        x, weight = ctx.saved_tensors
        B, D, H = x.shape
        
        # Allocate tensors for gradients.
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(weight)

        # Define the grid dimensions for the backward kernel.
        grid = (B, D)
        RMSNorm_Triton_bwd[grid](
            dx, dy, dw, x, weight, x.stride(0), x.stride(1), ctx.ep, H, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        
        return dx, dw