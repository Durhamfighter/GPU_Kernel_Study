#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

__global__ void weight_dequant_fp32(int8_t * q_weight, int q_weight_stride, float * weight, int weight_stride, float * scale, int num_cols){
    int row = blockIdx.x;
    int idx = threadIdx.x;
    float s = scale[row];
    for (int col = idx; col< num_cols; col+=blockDim.x){
        float w = (float)q_weight[row* q_weight_stride + col] * s;
        weight[row* weight_stride + col] = w;
    }
}

int main() {
    int num_rows=360;
    int num_cols = 540; 
    int weight_stride= num_cols;
    int q_weight_stride = num_cols;

    size_t weight_size= num_rows * num_cols * sizeof(float);
    size_t q_weight_size= num_rows * num_cols * sizeof(int8_t);
    size_t scale_size=num_rows * sizeof(float);

    //Host
    int8_t *h_q_weight =(int8_t *)malloc(q_weight_size);
    float *h_weight=(float *)malloc(weight_size);
    float *h_scale= (float *)malloc(scale_size);

    // h_q_weight and h_scale initialization
    for (int r = 0; r < num_rows; r++) {
        h_scale[r] = (float)(r % 100 + 1) * 0.01f;
        for (int c = 0; c < num_cols; c++) {
            int q_val = (r - c) % 127;
            if (q_val > 127) q_val = 127;
            if (q_val < -127) q_val = -127;
            h_q_weight[r*num_cols+ c] = (int8_t)q_val;
        }
    }


    //device
    float *d_weight, *d_scale;
    int8_t *d_q_weight;
    cudaMalloc((void **)&d_weight, weight_size);
    cudaMalloc((void **)&d_q_weight, q_weight_size);
    cudaMalloc((void **)&d_scale, scale_size);
    cudaMemcpy(d_q_weight, h_q_weight, q_weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, scale_size, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim(num_rows);

    weight_dequant_fp32<<<gridDim, blockDim>>>(d_q_weight,q_weight_stride, d_weight, weight_stride,d_scale,num_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_weight, d_weight, weight_size, cudaMemcpyDeviceToHost);
    printf("First row, first 10 dequantized values:\n");
    for (int i = 0; i < 10; i++) {
        printf("weight[%d] = %f\n", i, h_weight[i]);
    }
    cudaFree(d_weight);
    cudaFree(d_q_weight);
    cudaFree(d_scale);
    free(h_weight);
    free(h_q_weight);
    free(h_scale);

    return 0;
}

