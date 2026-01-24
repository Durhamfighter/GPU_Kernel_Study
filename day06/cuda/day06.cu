#include <cuda_runtime.h>
#include <stdio.h>

__global__ void leaky_relu(float * x, float alpha, int num_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx <num_elements){
        x[idx] = x[idx]< 0.0f ? alpha * x[idx] : x[idx];
    }
}

int main(){
    const int num_elements = 1000;
    const int size = num_elements * sizeof(float);
    const float alpha = 3.0;

    float *h_x = (float *)malloc(size);
    
    for (int i = 0; i<num_elements; i++){
        h_x[i] = i-500;
    }

    float *d_x;
    cudaMalloc((void **)&d_x,size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    dim3 blockDim(256,1,1);
    dim3 gridDim((num_elements+blockDim.x-1)/blockDim.x,1,1);
    for (int i = 495; i < 505; i++) {
        printf(" before h_x[%d] = %f\n", i, h_x[i]);
    }
    leaky_relu<<<gridDim,blockDim>>>(d_x,alpha,num_elements);
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    for (int i = 495; i < 505; i++) {
        printf("h_x[%d] = %f\n", i, h_x[i]);
    }

    cudaFree(d_x);
    free(h_x);

    return 0;
}