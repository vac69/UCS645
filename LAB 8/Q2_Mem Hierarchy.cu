#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "Error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

#define TPB 256
#define SIZE (1<<20)

__global__ void reduceSimple(const float *in, float *out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) total += in[i];
        out[0] = total;
    }
}

__global__ void reduceShared(const float *in, float *out, int n) {
    __shared__ float buf[TPB];
    int t = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + t;
    buf[t] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (t < step) buf[t] += buf[t + step];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = buf[0];
}

__global__ void reduceWarp(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? in[idx] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, val);
}

__global__ void conflictTest(float *out, int stride, int n) {
    __shared__ float sm[1024];
    int t = threadIdx.x;
    sm[(t * stride) % 1024] = t;
    __syncthreads();
    if (t < n) out[t] = sm[(t * stride) % 1024];
}

int main() {
    printf("\n=== Parallel Reduction Benchmark ===\n");

    float *d_in, *d_mid, *d_out;
    CHECK(cudaMalloc(&d_in, SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_mid, ((SIZE + TPB - 1) / TPB) * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemset(d_in, 0, SIZE * sizeof(float)));

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    float time_ms;

    printf("%-18s %12s %12s\n", "Method", "Time(us)", "GB/s");
    printf("---------------------------------------------\n");

    CHECK(cudaEventRecord(s));
    reduceSimple<<<1,1>>>(d_in, d_out, SIZE);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&time_ms, s, e));
    printf("%-18s %12.1f %12.2f\n", "Naive", time_ms*1000, 0.0);

    int blocks = (SIZE + TPB - 1) / TPB;

    CHECK(cudaEventRecord(s));
    reduceShared<<<blocks, TPB>>>(d_in, d_mid, SIZE);
    reduceShared<<<1, TPB>>>(d_mid, d_out, blocks);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&time_ms, s, e));
    float bw = (SIZE * sizeof(float)) / (time_ms / 1000.0) / 1e9;
    printf("%-18s %12.1f %12.2f\n", "Shared", time_ms*1000, bw);

    CHECK(cudaEventRecord(s));
    CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduceWarp<<<blocks, TPB>>>(d_in, d_out, SIZE);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&time_ms, s, e));
    float bw2 = (SIZE * sizeof(float)) / (time_ms / 1000.0) / 1e9;
    printf("%-18s %12.1f %12.2f\n", "Warp", time_ms*1000, bw2);

    printf("\nStride Test:\n");
    printf("%8s %12s\n", "Stride", "Time(us)");

    int strides[] = {1,2,4,8,16,32};
    for (int i = 0; i < 6; i++) {
        int st = strides[i];
        CHECK(cudaEventRecord(s));
        conflictTest<<<1,1024>>>(d_in, st, SIZE);
        CHECK(cudaEventRecord(e));
        CHECK(cudaEventSynchronize(e));
        CHECK(cudaEventElapsedTime(&time_ms, s, e));
        printf("%8d %12.1f\n", st, time_ms*1000);
    }

    printf("\nExecution finished.\n");

    cudaFree(d_in);
    cudaFree(d_mid);
    cudaFree(d_out);
    return 0;
}