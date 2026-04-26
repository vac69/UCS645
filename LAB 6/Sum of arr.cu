#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel to compute sum using atomic operation
__global__ void computeSum(const float *inputArr, float *result, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total) {
        atomicAdd(result, inputArr[tid]);
    }
}

int main() {
    int numElements = 1 << 20;   // 1048576 elements
    size_t bytes = numElements * sizeof(float);

    // Host allocations
    float *hostData = (float *)malloc(bytes);
    float *hostResult = (float *)malloc(sizeof(float));

    // Initialize data (all ones)
    for (int i = 0; i < numElements; ++i) {
        hostData[i] = 1.0f;
    }

    // Device pointers
    float *devData = NULL;
    float *devResult = NULL;

    // Allocate memory on GPU
    cudaMalloc((void **)&devData, bytes);
    cudaMalloc((void **)&devResult, sizeof(float));

    // Copy input to device
    cudaMemcpy(devData, hostData, bytes, cudaMemcpyHostToDevice);

    // Initialize result to zero
    cudaMemset(devResult, 0, sizeof(float));

    // Configure execution parameters
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Launch kernel
    computeSum<<<gridSize, blockSize>>>(devData, devResult, numElements);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hostResult, devResult, sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(devData);
    cudaFree(devResult);

    // Output result
    printf("Computed Sum = %.2f (Expected: %d)\n", *hostResult, numElements);

    // Cleanup host memory
    free(hostData);
    free(hostResult);

    return 0;
}