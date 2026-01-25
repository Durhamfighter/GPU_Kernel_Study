#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void fused_softmax(float *x, float *y, int n_rows, int n_cols){
    // Grid 1차원 Block 1차원 blockIdx.x -> next row
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int col = threadIdx.x; 
    float local_max = -INFINITY;
    for (int col_start = col; col_start<n_cols; col_start += BlockDim.x){
        if (local_max<x[row*n_cols+col_start]){
            local_max=x[row*n_cols+col_start];
        }
    }
    sdata[col] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset = offset/2) {
    if (col < offset)
        sdata[col] = fmaxf(sdata[col], sdata[col + offset]);
    __syncthreads();
    }

    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int c = col; c < n_cols; c += blockDim.x) {
        local_sum += expf(x[row * n_cols + c] - row_max);
    }

    sdata[col] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (col < offset)
            sdata[col] += sdata[col + offset];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();
    for (int c = col; c < n_cols; c += blockDim.x) {
        y[row * n_cols + c] =
            expf(x[row * n_cols + c] - row_max) / row_sum;
    }
}

int main() {
    // matrix shape
    int n_rows = 352;
    int n_cols = 1000;
    int num_elements = n_rows * n_cols;
    int size = num_elements * sizeof(float);

    // host variable
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);

    // input 초기화
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            h_x[r * n_cols + c] = (float)(1000 - c);
        }
    }

    // device variable
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim(n_rows, 1, 1);

    int shared_mem_size = blockDim.x * sizeof(float);

    fused_softmax<<<gridDim, blockDim,shared_mem_size>>>(
        d_x, d_y, n_rows, n_cols
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

   
    float sum = 0.0f;
    for (int c = 0; c < n_cols; c++)
        sum += h_y[c];

    printf("row 0 softmax sum = %f \n", sum);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
