#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024

__global__ void sumAtomic(const int *in, int *out) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        atomicAdd(&out[0], in[id]);
    }
}

__global__ void sumFormula(int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int result = SIZE * (SIZE + 1) / 2;
        out[1] = result;
    }
}

int main() {
    int host_in[SIZE];
    int host_out[2] = {0};

    for (int i = 0; i < SIZE; ++i) {
        host_in[i] = i + 1;
    }

    int *dev_in, *dev_out;

    cudaMalloc((void**)&dev_in, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_out, 2 * sizeof(int));

    cudaMemcpy(dev_in, host_in, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_out, 0, 2 * sizeof(int));

    int tpb = 256;
    int bpg = (SIZE + tpb - 1) / tpb;

    sumAtomic<<<bpg, tpb>>>(dev_in, dev_out);
    sumFormula<<<1, 1>>>(dev_out);

    cudaMemcpy(host_out, dev_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Iterative Sum (1 to %d) = %d\n", SIZE, host_out[0]);
    printf("Formula Sum (1 to %d)   = %d\n", SIZE, host_out[1]);
    printf("Results should be identical.\n");

    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}