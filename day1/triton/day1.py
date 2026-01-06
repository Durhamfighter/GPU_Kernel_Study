# triton day1
import triton 
import triton.language as tl
import torch


@triton.jit
def add_kernel(x_ptr,y_ptr,z_ptr,n_elements,BLOCK_SIZE:tl.constexpr):
    block_idx = tl.program_id(0)

    block_start = block_idx * BLOCK_SIZE

    offsets = block_start + tl.arange(0,BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets,mask=mask)
    y = tl.load(y_ptr + offsets,mask=mask)
    z = x + y

    tl.store(z_ptr + offsets,z,mask=mask)

if __name__ == "__main__":
    n_elements = 1000
    BLOCK_SIZE = 256

    x = torch.randn(n_elements,dtype=torch.float32,device='cuda')
    y = torch.randn(n_elements,dtype=torch.float32,device='cuda')
    z = torch.zeros(n_elements,dtype=torch.float32,device='cuda')
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add_kernel[grid](x,y,z,n_elements,BLOCK_SIZE)

    print(z)