import triton
import triton.language as tl


import torch
@triton.jit
def subtract_kernel(x_ptr,y_ptr,z_ptr,block_size,num_elements):
    
    block_idx = tl.program_id(0)
    block_start = block_idx * block_size 
    offsets = block_start + tl.arange(0,block_size)
    mask = offsets<num_elements
    x= tl.load(x_ptr+offsets, mask=mask)
    y= tl.load(y_ptr+offsets, mask =mask)

    z= x-y
    
    tl.store(z_ptr+offsets, z, mask=mask)
    
if __name__ == "__main__":
    
    num_elements =1000
    block_size = 256
    x=torch.randn(num_elements,dtype=torch.bfloat16,device= "cuda")
    y=torch.randn(num_elements,dtype=torch.bfloat16,device= "cuda")
    z=torch.empty(num_elements,dtype=torch.bfloat16,device="cuda")
    grid = (((num_elements+block_size-1)//block_size),)
    
    subtract_kernel[grid](x,y,z,block_size,num_elements)
    
    print(z)
    

