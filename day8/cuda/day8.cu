#include <cuda_runtime.h>
#include <stdio.h>

constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;

__global__ void gelu_approx(float *x ,int num_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements){
        x[idx] = 0.5 * x[idx]*(1+tanhf(SQRT_2_OVER_PI*(x[idx]+0.044715*x[idx]*x[idx]*x[idx])));
    }
}

int main(){
    const int num_elements= 1000;
    const int size = num_elements * sizeof(float);
    float *h_x = (float *)malloc(size);

    for (int i = 0; i<num_elements; i++){
        h_x[i] = (i - 500) * 0.01f;  
    }

    float *d_x;
    cudaMalloc((void **)&d_x, size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);

    dim3 blockDim(256,1,1);
    dim3 gridDim((num_elements+blockDim.x-1)/blockDim.x,1,1);
    
    gelu_approx<<<gridDim, blockDim>>>(d_x, num_elements);
    cudaMemcpy(h_x,d_x,size,cudaMemcpyDeviceToHost);

    float min_x = 2;

    for (int i = 0; i < 1000; i++) {
        if (min_x>h_x[i]){
            min_x= h_x[i];
        }
    }
    printf("min_x: %f\n",min_x);

    cudaFree(d_x);
    free(h_x);

    return 0;

}