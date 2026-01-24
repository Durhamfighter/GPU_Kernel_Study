#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_average(float *x, float *block_sum ,int num_elements){
    __shared__ float smem[256];
    int tid = threadIdx.x ; 
    int idx = blockDim.x * blockIdx.x + tid; // 벡터 x에서의 idx
    
    smem[tid] = (idx < num_elements) ? x[idx] : 0.0f; // smem 에 저장
    __syncthreads(); // thread syncronization

    for (int s = blockDim.x/2; s>0; s=s/2){
        if (tid<s){
            smem[tid] += smem[tid+s];
        }
        __syncthreads();
    } 
    // step 1: 256 → 128
    // step 2: 128 →  64
    // step 3:  64 →  32
    // step 4:  32 →  16
    // step 5:  16 →   8
    // step 6:   8 →   4
    // step 7:   4 →   2
    // step 8:   2 →   1 
    // 의문점 : step 5(16->8) 부터는 하나의 warp 중 절반만 일하니까 결국 비효율적이아닌가?
    if (tid == 0){
        block_sum[blockIdx.x] = smem[0];
    }
}

int main(){
    int num_elements = 1000;
    int size = num_elements * sizeof(float);


    // declare host variable
    
    float *h_x = (float *)malloc(size);
    for (int i = 0; i<num_elements; i++){
        h_x[i] = 500 - i;
    }
    
    // declare device variable
    float *d_x;
    cudaMalloc((void **)&d_x,size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);

    // Block, Grid size 정의
    dim3 blockDim(256,1,1);
    dim3 gridDim((num_elements+blockDim.x-1)/blockDim.x,1,1);

    // block 갯수 정의
    float *block_count = (float *)malloc(gridDim.x * sizeof(float));
    for (int i= 0; i<gridDim.x; i++){
        if (i != gridDim.x - 1)
            block_count[i] = blockDim.x;
        else {
            int last = num_elements % blockDim.x;
            block_count[i] = (last == 0) ? blockDim.x : last;
        }
    }

    //  block 별 summation 정의
    float *block_sum;
    float *h_block_sum = (float *)malloc(gridDim.x * sizeof(float));

    cudaMalloc((void **)&block_sum,gridDim.x*sizeof(float));
    cudaMemset(block_sum, 0, gridDim.x * sizeof(float));

    //커널 실행
    vector_average<<<gridDim,blockDim>>>(d_x,block_sum,num_elements);

    cudaMemcpy(h_block_sum, block_sum, gridDim.x * sizeof(float), cudaMemcpyDeviceToHost);

    // 아웃풋 구하기
    float total = 0.0f;
    float count = 0.0f;
    for (int i= 0; i<gridDim.x; i++){

        total += h_block_sum[i];
        count += block_count[i];
    }
    float global_mean = total / count;
    printf("global_mean: %f\n",global_mean);
    cudaFree(d_x);
    free(h_x);
}
