import triton
import triton.language as tl 
import torch 

@triton.jit
def tanh(x_ptr,num_elements,BLOCK_SIZE : tl.constexpr):
    
    # whenever you write triton, you should make sure that you write down kernel first and then decide grid size
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets <num_elements
    
    x = tl.load(x_ptr + offsets, mask = mask)
    x = 2*tl.sigmoid(2*x) - 1
    tl.store(x_ptr+offsets,x,mask = mask)
    
if __name__ == "__main__":
    num_elements = 1000
    BLOCK_SIZE = 256
    
    x= torch.randn(num_elements,device = 'cuda')
    
    grid = ((num_elements+BLOCK_SIZE-1)//BLOCK_SIZE,)
    
    tanh[grid](x,num_elements,BLOCK_SIZE)
    
    print(min(x),max(x))