import triton 
import triton.language as tl 
import torch 

@triton.jit
def MatMul(x_ptr, y_ptr, output_ptr, M, N, K, 
           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, b_k: tl.constexpr):
    
    #  program_id of block
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    
    #  offsets for 1 D
    offsets_row = row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_col = col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offsets_k =  tl.arange(0, b_k)
    a_offsets = offsets_row[:, None] * K + offsets_k[None, :]
    b_offsets = offsets_k[:, None] * N + offsets_col[None, :]

    # compute A[M,K] * B[K,N]
    for i in range(0, K, b_k):
        a_mask = (offsets_row[:,None]<M) & (i+offsets_k[None,:]<K)
        b_mask = (i+offsets_k[:, None] < K) & (offsets_col[None, :] < N)
        
        # computation
        a = tl.load(x_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(y_ptr + b_offsets, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_offsets += b_k * 1
        b_offsets += b_k * N
    
    # store results
    c_offsets = offsets_row[:, None] * N + offsets_col[None, :]
    c_mask = (offsets_row[:, None] < M) & (offsets_col[None, :] < N)
    tl.store(output_ptr + c_offsets, acc, mask=c_mask)
        