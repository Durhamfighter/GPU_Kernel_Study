#include <cuda_runtime.h>
#include <stdio.h>

__global__ void tanh(float *x,int num_elements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements){
        x[idx]=2/(1+exp(-2*x[idx])) -1;
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

    dim3 blockDim(256,1,1);
    dim3 gridDim((num_elements+blockDim.x-1)/blockDim.x,1,1);
    
    tanh<<<gridDim,blockDim>>>(d_x,num_elements);
    cudaMemcpy(h_x,d_x,size,cudaMemcpyDeviceToHost);

    float min_x = 100;
    float max_x = -100;

    for (int i = 0; i<num_elements; i++){
        if (min_x >h_x[i]){
            min_x = h_x[i];
        }
        if (max_x<h_x[i]){
            max_x = h_x[i];
        }
    }
    printf("min_x: %f\n",min_x);
    printf("max_x: %f\n",max_x);

    cudaFree(d_x);
    free(h_x);

}