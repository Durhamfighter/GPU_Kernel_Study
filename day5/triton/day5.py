import triton
import triton.language as tl 
import torch 

@triton.jit
def Relu(x_ptr,num_elements,BLOCK_SIZE : tl.constexpr):
    
    block_idx = tl.program_id(axis = 0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elements
    
    x = tl.load(x_ptr + offsets,mask = mask)
    x = tl.where(x>=0,x,0)
    tl.store(x_ptr+offsets,x , mask = mask)
    
if __name__ == "__main__":
    
    num_elements = 1000
    BLOCK_SIZE = 256
    
    x = torch.randn(num_elements,dtype=torch.bfloat16,device="cuda")
    grid = ((num_elements+BLOCK_SIZE -1 )//BLOCK_SIZE,)
    Relu[grid](x,num_elements,BLOCK_SIZE)
    
    print(x)
    
