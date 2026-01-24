import triton 
import triton.language as tl 
import torch

@triton.jit
def vector_average(x_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_idx = tl.program_id(axis=0)
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)
    block_count = tl.sum(mask, axis=0)

    avg = block_sum / block_count
    tl.store(output_ptr + block_idx, avg)

if __name__ =="__main__":
    num_elements = 1000
    BLOCK_SIZE = 256
    x= torch.randn(num_elements,device = 'cuda')
    num_blocks = (num_elements+BLOCK_SIZE-1)//BLOCK_SIZE
    output = torch.ones(num_blocks,device = 'cuda')
    
    grid =((num_elements+BLOCK_SIZE-1)//BLOCK_SIZE,)
    vector_average[grid](x,output,num_elements,BLOCK_SIZE) # x_output = [3213,]
    block_counts = torch.full((num_blocks,),BLOCK_SIZE,device='cuda')
    last =num_elements%BLOCK_SIZE
    if last != 0 :
        block_counts[-1] = last
    global_mean = (output * block_counts).sum() / block_counts.sum()
    print(global_mean)