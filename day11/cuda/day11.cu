#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void Transpose_2D(float *x, float *output, int M, int N, int BLOCK_SIZE_M, int BLOCK_SIZE_N) {
    int row_idx = blockIdx.y;
    int col_idx = blockIdx.x;
    
    int row_offset = row_idx * BLOCK_SIZE_M + threadIdx.y;
    int col_offset = col_idx * BLOCK_SIZE_N + threadIdx.x;
    
    if (row_offset < M && col_offset < N) {
        int input_idx = row_offset * N + col_offset;
        float value = x[input_idx];
        
        int output_row = col_offset;
        int output_col = row_offset;
        int output_idx = output_row * M + output_col;
        
        if (output_row < N && output_col < M) {
            output[output_idx] = value;
        }
    }
}

int main() {
    int M = 1000;
    int N = 800;
    int BLOCK_SIZE_M = 32;
    int BLOCK_SIZE_N = 16;
    int size = M * N * sizeof(float);
    
    float *h_x = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    for (int i = 0; i< M; i++) {
        for (int j= 0; j< N; j++) {
            h_x[i * N + j]=i-j;
        }
    }
    
    float *d_x;
    float *d_output;
    
    cudaMalloc((void **)&d_x,size);
    cudaMalloc((void **)&d_output,size);
    cudaMemcpy(d_x, h_x,size, cudaMemcpyHostToDevice);
    
    int num_row=(M +BLOCK_SIZE_M-1)/BLOCK_SIZE_M;
    int num_col = (N +BLOCK_SIZE_N- 1)/BLOCK_SIZE_N;
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M, 1);
    dim3 gridDim(num_col, num_row, 1);
    
    Transpose_2D<<<gridDim, blockDim>>>(d_x, d_output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N);
    
    cudaMemcpy(h_output, d_output,size, cudaMemcpyDeviceToHost);
    
    printf("Original[0][1] = %f, Transposed[1][0] = %f\n", h_x[1], h_output[1*M]);
    cudaFree(d_x);
    cudaFree(d_output);
    
    free(h_x);
    free(h_output);
    
    return 0;
}