#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__ void dropout_rgn(float *x, int num_elements, float p , int seed){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= num_elements) return ; 
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float r = curand_uniform(&state);

    if (r>p){
        x[idx] = x[idx] / (1.0f-p);
    }
    else{
        x[idx] = 0.0f;
    }
}
int main(){
    // matrix shape
    int num_elements = 352;
    int size = num_elements * sizeof(float);

    // host variable
    float *h_x = (float *)malloc(size);

    // input 초기화
    for (int r = 0; r < num_elements; r++) {
            h_x[r] = (float)(100 - r);
        }

    // device variable
    float *d_x;
    cudaMalloc((void **)&d_x, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim((num_elements + blockDim.x - 1)/blockDim.x, 1, 1);
    float p = 0.4;
    int seed = 32;

    dropout_rgn<<<gridDim, blockDim>>>(
        d_x, num_elements, p, seed
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

    // print first row
    for (int c = 0; c < num_elements; c++) {
        printf("h_x[%d] = %f\n", c, h_x[c]);
    }

    cudaFree(d_x);
    free(h_x);
    return 0;
}