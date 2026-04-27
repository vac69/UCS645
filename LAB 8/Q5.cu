#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        printf("CUDA failure: %s\n", cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void forwardStep(float *arr, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        arr[id] = arr[id] * 0.99f + 0.01f;
    }
}

int main() {
    printf("\n=============================================\n");
    printf(" CUDA Training Pipeline Simulation\n");
    printf("=============================================\n");

    cudaDeviceProp gpu;
    CHECK_CUDA(cudaGetDeviceProperties(&gpu, 0));
    printf("Device: %s\n\n", gpu.name);

    int total = 1 << 20;
    float *deviceData;

    CHECK_CUDA(cudaMalloc(&deviceData, total * sizeof(float)));
    CHECK_CUDA(cudaMemset(deviceData, 0, total * sizeof(float)));

    int tpb = 256;
    int bpg = (total + tpb - 1) / tpb;

    printf("Epoch | Loss    | Accuracy\n");
    printf("---------------------------\n");

    float currentLoss = 2.3f;
    float currentAcc  = 20.0f;

    for (int e = 1; e <= 10; e++) {

        forwardStep<<<bpg, tpb>>>(deviceData, total);
        CHECK_CUDA(cudaDeviceSynchronize());

        currentLoss *= 0.75f;
        currentAcc += (95.0f - currentAcc) * 0.25f;

        printf("%5d | %.4f | %.2f%%\n", e, currentLoss, currentAcc);
    }

    printf("\nPipeline execution complete.\n");
    printf("Computation simulated successfully.\n");

    cudaFree(deviceData);
    return 0;
}