#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DIM 1024   // Matrix dimension (1024 x 1024)

// Kernel for element-wise matrix addition
__global__ void addMatrices(const int *matA, const int *matB, int *matC, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < n && y < n) {
        int position = y * n + x;
        matC[position] = matA[position] + matB[position];
    }
}

int main() {
    int totalSize = DIM * DIM;
    size_t memBytes = totalSize * sizeof(int);

    // Host memory allocation
    int *hostA = (int *)malloc(memBytes);
    int *hostB = (int *)malloc(memBytes);
    int *hostC = (int *)malloc(memBytes);

    // Initialize input matrices
    for (int i = 0; i < totalSize; ++i) {
        hostA[i] = i % 100;
        hostB[i] = i % 50;
    }

    // Device pointers
    int *devA = NULL, *devB = NULL, *devC = NULL;

    // Allocate GPU memory
    cudaMalloc((void **)&devA, memBytes);
    cudaMalloc((void **)&devB, memBytes);
    cudaMalloc((void **)&devC, memBytes);

    // Transfer data to GPU
    cudaMemcpy(devA, hostA, memBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, memBytes, cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((DIM + blockDim.x - 1) / blockDim.x,
                 (DIM + blockDim.y - 1) / blockDim.y);

    // Kernel launch
    addMatrices<<<gridDim, blockDim>>>(devA, devB, devC, DIM);

    // Copy results back to host
    cudaMemcpy(hostC, devC, memBytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    // Output check
    printf("Matrix addition done.\n");
    printf("First element result: %d\n", hostC[0]);

    // Free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}