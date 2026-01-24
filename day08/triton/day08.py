import triton
import triton.language as tl 
import torch
from triton.language import math


@triton.jit
def gelu_approx(x_ptr,num_elements,BLOCK_SIZE:tl.constexpr):
    
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets<num_elements
    
    x = tl.load(x_ptr+offsets,mask=mask)
    sqrt_2_over_pi = 0.7978845608028654
    x = 0.5*x*(1+2 * tl.sigmoid(2 * sqrt_2_over_pi*(x+0.044715*x*x*x)) - 1)
    tl.store(x_ptr+offsets,x,mask =mask )
    
if __name__ == "__main__":
    
    BLOCK_SIZE = 256
    num_elements = 1000
    x = torch.randn(num_elements,dtype=torch.float32,device='cuda')
    grid=((num_elements+BLOCK_SIZE-1)//BLOCK_SIZE,)
    gelu_approx[grid](x,num_elements,BLOCK_SIZE)
    print(x)
    