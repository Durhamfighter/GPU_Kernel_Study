import triton 
import triton.language as tl 
import torch

@triton.jit
def weight_dequant_fp32(q_weight_ptr,q_weight_stride,weight_ptr,weight_stride,num_cols,scale_ptr,BLOCK_SIZE : tl.constexpr):
    row = tl.program_id(axis=0)
    col_off = tl.arange(0,BLOCK_SIZE)
    mask = col_off<num_cols
    q_weight = tl.load(q_weight_ptr + row*q_weight_stride+col_off , mask = mask)
    scale = tl.load(scale_ptr + row)
    weight = q_weight.to(tl.float32) * scale
    tl.store(weight_ptr + row*weight_stride + col_off,weight,mask = mask)

if __name__=='__main__':
    q_weight = torch.randint(-127,128,(352,1000),dtype=torch.int8,device='cuda')
    scale = torch.rand(352,dtype=torch.float32,device='cuda') * 0.1 + 0.01
    weight = torch.empty(352,1000,dtype=torch.float32,device='cuda')
    row,col = q_weight.shape
    BLOCK_SIZE = triton.next_power_of_2(col)
    
    grid = (row,1,1)
    weight_dequant_fp32[grid](q_weight,q_weight.stride(0),weight,weight.stride(0), col,scale,BLOCK_SIZE)
    print(weight)

