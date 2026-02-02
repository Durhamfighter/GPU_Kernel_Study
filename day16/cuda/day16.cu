#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dropout_fixed(float *x,int x_stride,int n_cols,int col_to_dropout){
    int row = blockIdx.x;
    int idx = threadIdx.x;
    for (int col = idx; col < n_cols; col += blockDim.x ){
        if (col == col_to_dropout){
            x[row * x_stride + col] =0;
        }
    }
}

int main(){
    // matrix shape
    int n_rows = 352;
    int n_cols = 1000;
    int num_elements = n_rows * n_cols;
    int size = num_elements * sizeof(float);

    // host variable
    float *h_x = (float *)malloc(size);

    // input 초기화
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            h_x[r * n_cols + c] = (float)(1000 - c);
        }
    }

    // device variable
    float *d_x;
    cudaMalloc((void **)&d_x, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim( n_rows, 1, 1);
    int col_to_dropout = 0;
    dropout_fixed<<<gridDim, blockDim>>>(
        d_x, n_cols,n_cols,col_to_dropout
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

    // print first row
    for (int c = 0; c < n_cols; c++) {
        printf("h_x[%d] = %f\n", c, h_x[c]);
    }

    cudaFree(d_x);
    free(h_x);
    return 0;
}