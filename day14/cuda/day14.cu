#include <cuda_runtime.h>
#include <stdio.h>
__global__ void l2norm(float *x, float *y, int n_rows, int n_cols){
    // Grid 1차원 Block 1차원 blockIdx.x -> next row
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int col = threadIdx.x; 
    float cul_sum = 0.0f;
    for (int col_start = col; col_start<n_cols; col_start += blockDim.x){
            cul_sum+=x[row*n_cols+col_start]*x[row*n_cols+col_start];
        }
    sdata[col] = cul_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset = offset/2) {
    if (col < offset){
        sdata[col] = sdata[col] + sdata[col + offset];
    }
    __syncthreads();
    }
    if (col == 0)
        y[row] = sqrtf(sdata[0]);
    __syncthreads();
}

int main(){
    // matrix shape
    int n_rows = 352;
    int n_cols = 1000;
    int num_elements = n_rows * n_cols;
    int size = num_elements * sizeof(float);
    int y_size = n_rows * sizeof(float);

    // host variable
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(n_rows * sizeof(float));

    // input 초기화
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            h_x[r * n_cols + c] = (float)(1000 - c);
        }
    }

    // device variable
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, y_size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim(n_rows, 1, 1);

    int shared_mem_size = blockDim.x * sizeof(float);

    l2norm<<<gridDim, blockDim,shared_mem_size>>>(
        d_x, d_y, n_rows, n_cols
    );

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, y_size, cudaMemcpyDeviceToHost);;

    for (int c = 0; c < n_rows; c++)
        printf("h_y[%d] : %f\n", c, h_y[c]);

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}