import triton 
import triton.language as tl 
import torch 

@triton.jit
def Transpose_2D(x_ptr,output_ptr,M,N,BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_N : tl.constexpr):
    
    block_idx = tl.program_id(0)
    num_col = tl.cdiv(N, BLOCK_SIZE_N)
    num_row = tl.cdiv(M, BLOCK_SIZE_M)
    col_idx = block_idx % num_col
    row_idx=  block_idx // num_col

    
    row_offset = row_idx * BLOCK_SIZE_M +tl.arange(0,BLOCK_SIZE_M)
    mask_row = row_offset<M
    col_offset = col_idx * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
    mask_col = col_offset<N
    
    offset_2D = row_offset[:,None]*N + col_offset[None,:]
    mask_2D = mask_row[:,None] & mask_col[None,:]
    
    x = tl.load(x_ptr+offset_2D,mask =mask_2D)
    offset_2D_transpose = col_offset[:,None] * M + row_offset[None,:]
    mask_2D_transpose =  mask_col[:, None] & mask_row[None,:] 
    #TODO Optimization needs for memory coalescing in store
    # The next time we need to study about tl.trans(X^T)
    x_t = tl.trans(x)
    tl.store( output_ptr + offset_2D_transpose, x_t,mask = mask_2D_transpose)
    
    
if __name__ == '__main__':
    M,N = 1000,800
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 16
    x = torch.randn((M,N),device = 'cuda')
    y = torch.ones((N,M),device ='cuda')
    
    num_row = triton.cdiv(M, BLOCK_SIZE_M)
    num_col = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_row * num_col,)
    Transpose_2D[grid](x,y,M,N,BLOCK_SIZE_M,BLOCK_SIZE_N)
    print(y)