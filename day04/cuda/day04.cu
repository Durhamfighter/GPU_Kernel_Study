#include <cuda_runtime.h>
#include <stdio.h>


__global__ void fused_axpby(float *x, float *y, const float alpha, const float beta,int n_elements){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx<n_elements){
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}



int main(){

    const int n_elements = 1000;
    const int size = n_elements * sizeof(float);

    const float alpha = 2.0;
    const float beta = 3.0;


    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);

    for (int i=0; i<n_elements; i++){
        h_x[i] = 1;
        h_y[i] = 1;
    }

    float *d_x,*d_y;

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);


    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);

    dim3 blockDim(256,1,1);
    dim3 gridDim((n_elements+blockDim.x-1)/blockDim.x,1,1);

    fused_axpby<<<gridDim,blockDim>>>(d_x,d_y,alpha,beta,n_elements);

    cudaDeviceSynchronize();
    cudaMemcpy(h_y,d_y,size,cudaMemcpyDeviceToHost);
    
    printf("h_y: %f\n",h_y[3]);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
    
}