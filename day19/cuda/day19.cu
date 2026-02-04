#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

__global__ void weight_quant_int8(float *weight,int weight_stride, int8_t * q_weight, int q_weight_stride, float * scale, int num_cols){
    int row = blockIdx.x;
    int idx = threadIdx.x;
    float s = scale[row];
    for (int col = idx; col< num_cols; col+=blockDim.x){
        float w = weight[row* weight_stride + col]/s;
        int q = (int)lrintf(w);
        q = q > 127 ? 127 : q;
        q = q < -127 ? -127 : q;
        q_weight[row* q_weight_stride + col] = (int8_t)q;
    }
}

int main() {
    int num_rows = 360;
    int num_cols = 540; 
    int weight_stride = num_cols;
    int q_weight_stride = num_cols;

    size_t weight_size = num_rows * num_cols * sizeof(float);
    size_t q_weight_size = num_rows * num_cols * sizeof(int8_t);
    size_t scale_size = num_rows * sizeof(float);

    //Host
    float *h_weight = (float *)malloc(weight_size);
    int8_t *h_q_weight = (int8_t *)malloc(q_weight_size);
    float *h_scale = (float *)malloc(scale_size);

    // h_weight intiailisation
    for (int r = 0; r < num_rows; r++) {
        float max_abs = 0.0f;
        for (int c = 0; c < num_cols; c++) {
            float w = (float)(r - c) * 0.05f;
            h_weight[r * num_cols + c] = w;
            float abs_w = fabsf(w);
            if (abs_w > max_abs) {
            max_abs = abs_w;
            }
            h_scale[r] = max_abs/127;
        }
    }


    //device
    float *d_weight, *d_scale;
    int8_t *d_q_weight;
    cudaMalloc((void **)&d_weight, weight_size);
    cudaMalloc((void **)&d_q_weight, q_weight_size);
    cudaMalloc((void **)&d_scale, scale_size);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, scale_size, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim(num_rows);

    weight_quant_int8<<<gridDim, blockDim>>>(d_weight,weight_stride, d_q_weight, q_weight_stride,d_scale,num_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_q_weight, d_q_weight, q_weight_size, cudaMemcpyDeviceToHost);
    printf("First row, first 10 quantized values:\n");
    for (int i = 0; i < 10; i++) {
        printf("q_weight[%d] = %d\n", i, h_q_weight[i]);
    }
    cudaFree(d_weight);
    cudaFree(d_q_weight);
    cudaFree(d_scale);
    free(h_weight);
    free(h_q_weight);
    free(h_scale);

    return 0;
}