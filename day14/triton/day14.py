import triton 
import trtion.language as tl 
import torch

@triton.jit
def l2_norm_rowwise(x_ptr,output_ptr,stride_x,N,BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row*stride_x
    col_offsets = tl.arange(0,BLOCK_SIZE)
    mask = col_offsets < N
    x =tl.load(x_ptr + row_start +col_offsets,mask = mask,other =0.0)
    partial_sum = tl.sum(x*x,axis=0)
    tl.store(output_ptr + row,tl.sqrt(partial_sum))
    
if __name__ =='__main__':
    x= torch.randn(102,1000,device='cuda')
    row,col = x.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    output = torch.empty(row, device='cuda')
    
    l2_norm_rowwise[(row,)](x,output,x.stride(0),col,BLOCK_SIZE)
    