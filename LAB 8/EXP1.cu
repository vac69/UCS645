#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define BASE_THREADS 256

/* ================================================================
 * VECTOR ADD KERNEL
 * ================================================================ */
__global__ void addVectors(const float *x, const float *y, float *z, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        z[id] = x[id] + y[id];
    }
}

void hostAdd(const float *x, const float *y, float *z, int n) {
    for (int i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

/* ================================================================
 * PART A: CPU vs GPU PERFORMANCE
 * ================================================================ */
void runSpeedTest() {
    int sizes[] = {10, 14, 18, 22, 26};

    printf("\n=== Speed Comparison (CPU vs GPU) ===\n");
    printf("%10s %12s %12s %12s %12s\n", "N", "CPU(ms)", "GPU(ms)", "Copy(ms)", "Gain");
    printf("---------------------------------------------------------------\n");

    for (int i = 0; i < 5; i++) {
        int n = 1 << sizes[i];
        size_t mem = n * sizeof(float);

        float *h1 = (float*)malloc(mem);
        float *h2 = (float*)malloc(mem);
        float *hout = (float*)malloc(mem);
        float *href = (float*)malloc(mem);

        for (int j = 0; j < n; j++) {
            h1[j] = (float)rand() / RAND_MAX;
            h2[j] = (float)rand() / RAND_MAX;
        }

        // CPU timing
        clock_t cstart = clock();
        hostAdd(h1, h2, href, n);
        double cpu_time = (clock() - cstart) * 1000.0 / CLOCKS_PER_SEC;

        float *d1, *d2, *dout;
        CHECK_CUDA(cudaMalloc(&d1, mem));
        CHECK_CUDA(cudaMalloc(&d2, mem));
        CHECK_CUDA(cudaMalloc(&dout, mem));

        cudaEvent_t e0, e1;
        CHECK_CUDA(cudaEventCreate(&e0));
        CHECK_CUDA(cudaEventCreate(&e1));

        // Host to Device
        CHECK_CUDA(cudaEventRecord(e0));
        CHECK_CUDA(cudaMemcpy(d1, h1, mem, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d2, h2, mem, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));

        float transfer_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&transfer_ms, e0, e1));

        int tpb = BASE_THREADS;
        int bpg = (n + tpb - 1) / tpb;

        // Kernel execution
        CHECK_CUDA(cudaEventRecord(e0));
        addVectors<<<bpg, tpb>>>(d1, d2, dout, n);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));

        float kernel_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, e0, e1));

        printf("%10d %12.2f %12.2f %12.2f %12.2f\n",
               n, cpu_time, kernel_ms, transfer_ms, cpu_time / kernel_ms);

        cudaFree(d1); cudaFree(d2); cudaFree(dout);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        free(h1); free(h2); free(hout); free(href);
    }
}

/* ================================================================
 * PART B: BLOCK SIZE TESTING
 * ================================================================ */
void analyzeBlocks() {
    int options[] = {64, 128, 256, 512, 1024};
    int n = 1 << 20;
    size_t mem = n * sizeof(float);

    printf("\n=== Kernel Launch Configuration Study ===\n");
    printf("%12s %12s %12s\n", "Threads", "Blocks", "Time(ms)");
    printf("---------------------------------------------\n");

    float *h1 = (float*)malloc(mem);
    float *h2 = (float*)malloc(mem);

    for (int i = 0; i < n; i++) {
        h1[i] = 1.0f;
        h2[i] = 2.0f;
    }

    float *d1, *d2, *dout;
    CHECK_CUDA(cudaMalloc(&d1, mem));
    CHECK_CUDA(cudaMalloc(&d2, mem));
    CHECK_CUDA(cudaMalloc(&dout, mem));

    CHECK_CUDA(cudaMemcpy(d1, h1, mem, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d2, h2, mem, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    for (int i = 0; i < 5; i++) {
        int threads = options[i];
        int blocks = (n + threads - 1) / threads;

        CHECK_CUDA(cudaEventRecord(e0));
        addVectors<<<blocks, threads>>>(d1, d2, dout, n);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));

        float t = 0;
        CHECK_CUDA(cudaEventElapsedTime(&t, e0, e1));

        printf("%12d %12d %12.3f\n", threads, blocks, t);
    }

    cudaFree(d1); cudaFree(d2); cudaFree(dout);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    free(h1); free(h2);
}

/* ================================================================
 * MAIN
 * ================================================================ */
int main() {
    printf("\n===========================================\n");
    printf(" CUDA Experiment: Performance Evaluation\n");
    printf("===========================================\n");

    cudaDeviceProp info;
    CHECK_CUDA(cudaGetDeviceProperties(&info, 0));
    printf("Device: %s | Compute Capability: %d.%d\n\n",
           info.name, info.major, info.minor);

    runSpeedTest();
    analyzeBlocks();

    printf("\nExecution complete. Data ready for analysis.\n");
    return 0;
}