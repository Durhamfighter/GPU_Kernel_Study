#include <cuda_runtime.h>
#include <stdio.h>

__global__ void silu(float *x, int num_elements){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < num_elements){
    x[idx]=x[idx]/(1+exp(-x[idx]));
   }
}

int main(){
    int num_elements = 1000;
    int size = num_elements * sizeof(float);
    float *h_x = (float *)malloc(size);
    
    for (int i = 0; i<num_elements; i++){
        h_x[i] = (500 - i) * 0.01f;
    }

    float *d_x;

    cudaMalloc((void **)&d_x,size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);

    dim3 blockDim(256,1,1);
    dim3 gridDim((num_elements+blockDim.x-1)/blockDim.x,1,1);

    silu<<<gridDim,blockDim>>>(d_x,num_elements);
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