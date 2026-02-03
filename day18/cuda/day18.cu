#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

__global__ void unpacked_int32(int32_t *x, int8_t *output, int num_elements){
    /// this kerne unapckes int 32  to int 4
    /// each thread in block unpacks int32 from x vector to 8 int4 type element, then saves them to output vector
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx>=num_elements) return;
    int32_t packed = x[idx];

    // pragam unroll does unroll for-loop at compile-time
    #pragma unroll
    for (int i =0; i<8; i++){
        int shift = i * 4;
        int8_t value = (packed>>shift)& 0xF;

        if (value>=8) {
            value -= 16;
        }
        output[idx*8+i] = value;
    }
}


int main(){
    int num_elements = 1000;
    int int32_size = num_elements * sizeof(int32_t);

    // this is the vector that will include in4 element but saves as int 8 type vector
    int int8_size = num_elements * 8 * sizeof(int8_t);

    int32_t *h_x = (int32_t *)malloc(int32_size);
    int8_t *h_output = (int8_t *) malloc(int8_size);

    for (int i =0; i<num_elements;i++){
        h_x[i] = (500 - i);
    }

    //devcie variable
    int32_t *d_x;
    cudaMalloc((void **)&d_x, int32_size);
    cudaMemcpy(d_x, h_x, int32_size, cudaMemcpyHostToDevice);
    int8_t *d_output;
    cudaMalloc((void **)&d_output, int8_size);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim((num_elements + blockDim.x - 1)/blockDim.x, 1, 1);


    unpacked_int32<<<gridDim,blockDim>>>(
        d_x,d_output,num_elements
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output,int8_size,cudaMemcpyDeviceToHost);

    printf("First 10 unpacked values:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %d\n", i, h_output[i]);
    }
    cudaFree(d_x);
    cudaFree(d_output);
    free(h_x);
    free(h_output);
}