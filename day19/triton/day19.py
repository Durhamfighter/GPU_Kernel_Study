import triton 
import triton.language as tl 
import torch

@triton.jit
def weight_quant_int8(weight_ptr,weight_stride,q_weight_ptr,q_weight_stride,num_cols,scale_ptr,BLOCK_SIZE : tl.constexpr):
    row = tl.program_id(axis=0)
    col_off = tl.arange(0,BLOCK_SIZE)
    mask = col_off<num_cols
    weight = tl.load(weight_ptr + row*weight_stride+col_off , mask = mask)
    scale = tl.load(scale_ptr + row)
    q_weight = tl.maximum(tl.minimum(tl.round(weight/scale),127),-127).to(tl.int8) # int 8
    tl.store(q_weight_ptr + row*q_weight_stride + col_off,q_weight,mask = mask)

if __name__=='__main__':
    weight = torch.randn(352,1000,dtype=torch.float32,dtype='device')
    scale = [max(abs(row))/127 for row in weight]
    q_weight = torch.empty(352,1000,dtype=torch.int8,dtype='device')
    row,col = weight.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    
    grid = (row,1,1)
    weight_quant_int8[grid](weight,q_weight,)