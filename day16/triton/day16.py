import triton
import triton.language as tl 
import torch 

@triton.jit
def dropout_fixed(x_ptr, x_stride,n_cols,col_to_dropout,BLOCK_SIZE : tl.constexpr):
    row = tl.program_id(axis = 0)
    row_start = row * x_stride 
    col_offset = tl.arange(0,BLOCK_SIZE)
    offset = row_start + col_offset 
    mask = col_offset< n_cols
    x=tl.load(x_ptr+offset, mask = mask)
    keep = (col_offset!=col_to_dropout) & mask 
    x = x * keep
    tl.store(x_ptr+offset,x,mask=mask)

if __name__=='__main__':
    
    x= torch.randn(100,1000,device='cuda')
    col_to_dropout = 10
    row,col = x.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    dropout_fixed[(row,)](x,x.stride(0),n_cols=col,col_to_dropout=col_to_dropout,BLOCK_SIZE=BLOCK_SIZE)
    print(x)
