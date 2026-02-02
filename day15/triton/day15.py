import triton 
import triton.language as tl 
import torch

@triton.jit
def clamp_min_max(x_ptr,stride_x,min,max,n_cols,BLOCK_SIZE):
    row =tl.program_id(0)
    row_start = row * stride_x
    col_offsets = tl.arange(0,BLOCK_SIZE)
    mask = col_offsets< n_cols 
    x = tl.load(x_ptr+row_start+col_offsets,mask= mask)
    x = tl.clamp(x,min,max)
    tl.store(x_ptr+row_start+col_offsets,x,mask= mask)

if __name__ =='__main__':
    x= torch.randn(102,1000,device='cuda')
    row,col = x.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    min,max = 0,1
    clamp_min_max[(row,)](x,x.stride(0),min,max,col,BLOCK_SIZE)
    print(x)
    
    
    