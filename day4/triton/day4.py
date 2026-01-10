import triton
import triton.language as tl
import torch 


@triton.jit
def fused_axpby(x_ptr,y_ptr,alpha,beta,num_elemtns, BLOCK_SIZE:tl.constexpr):
    block_idx=tl.program_id(0)
    block_start= block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elemtns
    
    x = tl.load(x_ptr+offsets,mask=mask)
    y = tl.load(y_ptr+offsets,mask=mask)
    y= alpha*x + beta*y
    tl.store(y_ptr+offsets,y,mask=mask)


if __name__ == "__main__" : 
    num_elements = 1000
    block_size = 256
    
    x = torch.ones((num_elements,), device='cuda')
    y = torch.ones((num_elements,), device='cuda')
    alpha = 2
    beta = 3
    print(y)
    grid = ((num_elements+block_size -1)//block_size,)
    fused_axpby[grid](x,y,alpha,beta,num_elements,BLOCK_SIZE = block_size)
    
    print(y)
    