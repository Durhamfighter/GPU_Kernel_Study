import triton
import triton.language as tl 
import torch

@triton.jit
def leaky_relu(x_ptr,alpha,num_elements,BLOCK_SIZE : tl.constexp):

    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elements
    
    x = tl.load(x_ptr+offsets,mask = mask)
    x =tl.where(x<0,alpha*x,x)
    tl.store(x_ptr + offsets,x,mask = mask)

if __name__ == "__main__":
    
    BLOCK_SIZE = 256
    alpha = 3
    num_elements = 1000

    x = torch.randn(num_elements,device = 'cuda')
    print("before : ", x)
    grid=((num_elements + BLOCK_SIZE -1)/BLOCK_SIZE,)
    leaky_relu[grid](x,alpha,num_elements,BLOCK_SIZE)
    print("after : ", x)