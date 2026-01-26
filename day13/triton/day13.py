import triton 
import triton.language as tl 
import torch 

@triton.jit
def fused_softmax(output_ptr,input_ptr,input_row_stride,output_row_stride,n_rows,n_cols,BLOCK_SIZE : tl.constexpr):
    row_start=tl.program_id(axis=0) # 각 prgoram 의 row 시작
    row_step = tl.num_programs(0) # grid 0차원 dimension size
    # tiling  this is not online softmax 
    for row_idx in tl.range(row_start,n_rows,row_step):
        
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0,BLOCK_SIZE)
        mask = col_offsets < n_cols
        row=tl.load(row_start_ptr +col_offsets, mask = mask,other=-float('inf'))
        row_max = tl.max(row,axis=0)
        numerator = tl.exp(row-row_max)
        denorminator = tl.sum(numerator)
        result = numerator/denorminator
        # write it on output_ptr
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, result, mask=mask)


if __name__ == "__main__":
    x=torch.randn(232, 352, device='cuda')
    n_rows,n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid=(n_rows,1,1)
    fused_softmax[grid](y,x,x.stride(0),y.stride(0),n_rows,n_cols,BLOCK_SIZE)
    print(y)
    