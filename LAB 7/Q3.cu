#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 262144

__device__ float dX[SIZE];
__device__ float dY[SIZE];
__device__ float dZ[SIZE];

__global__ void addKernel() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE) {
        dZ[id] = dX[id] + dY[id];
    }
}

int main() {
    float hX[SIZE], hY[SIZE], hZ[SIZE];

    for (int i = 0; i < SIZE; i++) {
        hX[i] = 1.0f;
        hY[i] = 2.0f;
    }

    cudaMemcpyToSymbol(dX, hX, sizeof(hX));
    cudaMemcpyToSymbol(dY, hY, sizeof(hY));

    cudaEvent_t tStart, tEnd;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tEnd);

    int tpb = 256;
    int bpg = (SIZE + tpb - 1) / tpb;

    cudaEventRecord(tStart);
    addKernel<<<bpg, tpb>>>();
    cudaEventRecord(tEnd);
    cudaEventSynchronize(tEnd);

    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, tStart, tEnd);

    cudaMemcpyFromSymbol(hZ, dZ, sizeof(hZ));

    printf("Execution time: %.3f ms\n\n", time_ms);

    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);

    float memClock = device.memoryClockRate / 1000000.0f;
    float theoBW = (memClock * device.memoryBusWidth * 2) / 8.0f;

    printf("Theoretical Bandwidth\n");
    printf("Clock: %.2f GHz\n", memClock);
    printf("Bus Width: %d bits\n", device.memoryBusWidth);
    printf("Bandwidth: %.2f GB/s\n\n", theoBW);

    size_t readBytes = 2 * SIZE * sizeof(float);
    size_t writeBytes = SIZE * sizeof(float);
    size_t total = readBytes + writeBytes;

    float sec = time_ms / 1000.0f;
    float measured = total / (sec * 1e9f);

    printf("Measured Bandwidth: %.2f GB/s\n", measured);
    printf("Efficiency: %.1f%%\n\n", (measured / theoBW) * 100.0f);

    printf("Result OK: %.1f\n", hZ[0]);

    cudaEventDestroy(tStart);
    cudaEventDestroy(tEnd);

    return 0;
}