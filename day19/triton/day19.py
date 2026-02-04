import triton 
import triton.language as tl 
import torch
def tl_round(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
@triton.jit
def weight_quant_int8(weight_ptr,weight_stride,q_weight_ptr,q_weight_stride,num_cols,scale_ptr,BLOCK_SIZE : tl.constexpr):
    row = tl.program_id(axis=0)
    col_off = tl.arange(0,BLOCK_SIZE)
    mask = col_off<num_cols
    weight = tl.load(weight_ptr + row*weight_stride+col_off , mask = mask)
    scale = tl.load(scale_ptr + row)
    q = tl.where(
        weight >= 0,
        tl.floor(weight + 0.5),
        tl.ceil(weight - 0.5),
    )
    q_weight = tl.maximum(tl.minimum(q,127),-127).to(tl.int8) # int 8
    tl.store(q_weight_ptr + row*q_weight_stride + col_off,q_weight,mask = mask)

if __name__=='__main__':
    weight = torch.randn(352,1000,dtype=torch.float32,device='cuda')
    scale = weight.abs().max(dim=1).values / 127.0
    q_weight = torch.empty(352,1000,dtype=torch.int8,device='cuda')
    row,col = weight.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    
    grid = (row,1,1)
    weight_quant_int8[grid](weight,weight.stride(0),q_weight,q_weight.stride(0), col,scale,BLOCK_SIZE)
    print(q_weight)