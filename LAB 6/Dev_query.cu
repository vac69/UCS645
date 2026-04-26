#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaDeviceProp deviceInfo;

    cudaGetDeviceCount(&deviceCount);
    printf("Total CUDA Devices Available: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaGetDeviceProperties(&deviceInfo, dev);

        printf("---- GPU %d : %s ----\n", dev, deviceInfo.name);

        printf("Compute Capability      : %d.%d\n", 
               deviceInfo.major, deviceInfo.minor);

        printf("Max Threads Dimensions  : (%d, %d, %d)\n",
               deviceInfo.maxThreadsDim[0],
               deviceInfo.maxThreadsDim[1],
               deviceInfo.maxThreadsDim[2]);

        printf("Max Grid Size           : (%d, %d, %d)\n",
               deviceInfo.maxGridSize[0],
               deviceInfo.maxGridSize[1],
               deviceInfo.maxGridSize[2]);

        printf("Global Memory           : %.2f GB\n",
               deviceInfo.totalGlobalMem / (1024.0 * 1024 * 1024));

        printf("Constant Memory         : %lu bytes\n",
               deviceInfo.totalConstMem);

        printf("Shared Memory/Block     : %lu bytes\n",
               deviceInfo.sharedMemPerBlock);

        printf("Warp Size               : %d\n",
               deviceInfo.warpSize);

        printf("Max Threads/Block       : %d\n",
               deviceInfo.maxThreadsPerBlock);

        printf("Double Precision        : %s\n\n",
               (deviceInfo.major >= 2) ? "Supported" : "Not Supported");
    }

    return 0;
}