#include <cuda_runtime.h>
#include <stdio.h>


__global__ void subtract_kernel(float *x,float*y,float*z, int n_elements){
    int idx = blockIdx.x* blockDim.x + threadIdx.x;
    if (idx<n_elements){
        z[idx]=x[idx] - y[idx];
    }
}




int main(){
    const int n_elements = 1000;
    const int size = n_elements * sizeof(float);

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_z = (float*)malloc(size);

    for (int i=0; i<1000; i++){
        h_x[i]=i;
        h_y[i]=i;
    }

    float *d_x,*d_y,*d_z;

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMalloc((void**)&d_z, size);

    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);

    dim3 blockDim(256,1,1);
    dim3 gridDim((n_elements+blockDim.x-1)/blockDim.x,1,1);

    subtract_kernel<<<gridDim,blockDim>>>(d_x,d_y,d_z,n_elements);

    cudaDeviceSynchronize();

   cudaMemcpy(h_z,d_z,size,cudaMemcpyDeviceToHost);

   for (int i =0; i<10; i++){
    printf("%f\n",h_z[i]);
   }

   //clean
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}