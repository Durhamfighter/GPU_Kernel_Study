import triton 
import triton.language as tl 
import torch

@triton.jit
def vector_average(x_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_idx = tl.program_id(axis=0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    partial=tl.sum(x)/BLOCK_SIZE

    tl.store(output_ptr+block_idx,partial)


if __name__ =="__main__":
    
    torch.arange()