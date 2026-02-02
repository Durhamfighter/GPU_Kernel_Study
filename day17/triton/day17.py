import triton
import triton.language as tl 
import torch 


@triton.jit
def dropout_rgn(x_ptr,n_elements,p,seed,BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(axis=0)
    block_start =block_idx *BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask )
    random = tl.rand(seed,offsets)
    x_keep = random > p
    output = tl.where(x_keep,x/(1-p),0.0)
    tl.store(x_ptr+offsets,output,mask = mask)

if __name__ == "__main__":
    x = torch.randn(size=(254, ), device='cuda')
    assert x.is_contiguous()
    BLOCK_SIZE = 128
    num_elements = x.numel()
    grid = ((num_elements+BLOCK_SIZE-1)//BLOCK_SIZE,)
    print("before", x)
    dropout_rgn[grid](x,num_elements,0.4,123,BLOCK_SIZE)
    
    print("after", x)

    
