import triton 
import triton.language as tl 
import torch 

@triton.jit 
def int32_unpack_bitwise(packed_ptr,output_ptr,num_elements, BLOCK_SIZE : tl.constexpr):
    
    # here packed_ptr is 1-D Tensor whose element is int32
    block_idx=tl.program_id(axis = 0)
    offsets = block_idx * BLOCK_SIZE +tl.arange(0,BLOCK_SIZE)
    mask = offsets<num_elements
    packed = tl.load(packed_ptr+offsets, mask =mask) # tl.int32
    
    # extract 8 int4 values from packed
    for i in range(8):
        shift = i*4
        value = ((packed >> shift) & 0xF).to(tl.int8) # for pytorch dtype support as int4 does not exist
        tl.store(output_ptr + offsets*8 +i ,value,mask = mask)
        
if __name__ == "__main__" :
    packed_int32 = torch.randint( low=-(2**31), high=2**31, size=(1000,), dtype=torch.int32, device='cuda')
    output = torch.empty(8000,dtype = torch.int8,device = 'cuda')
    num_elements = 1000
    BLOCK_SIZE = 256
    grid = ((num_elements + BLOCK_SIZE - 1)//BLOCK_SIZE,1,1)
    int32_unpack_bitwise[grid](packed_int32,output,num_elements,BLOCK_SIZE)
    print(packed_int32)
    print(output)
    
    