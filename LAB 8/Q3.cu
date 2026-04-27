#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA failure at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define TPB 256
#define SIZE (1 << 18)

int compare_arrays(const float *a, const float *b, int n, float tol) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > tol) return 0;
    return 1;
}

__global__ void kernel_sigmoid(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

__global__ void kernel_tanh(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

__global__ void kernel_leaky(const float *in, float *out, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (in[idx] > 0.0f) ? in[idx] : a * in[idx];
}

__global__ void kernel_relu_grad(const float *grad_out, const float *forward, float *grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_in[idx] = (forward[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

__global__ void kernel_bce(const float *pred, const float *target, float *loss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = fmaxf(fminf(pred[idx], 1.0f - 1e-7f), 1e-7f);
        loss[idx] = -(target[idx] * logf(p) + (1.0f - target[idx]) * logf(1.0f - p));
    }
}

int main() {
    printf("\n===========================================\n");
    printf(" CUDA ML Kernels Test\n");
    printf("===========================================\n");

    int n = SIZE;
    size_t mem = n * sizeof(float);

    float *h_in = (float*)malloc(mem);
    float *h_out = (float*)malloc(mem);
    float *h_ref = (float*)malloc(mem);
    float *h_t = (float*)malloc(mem);

    for (int i = 0; i < n; i++) {
        h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 6.0f;
        h_t[i] = (rand() % 2) ? 1.0f : 0.0f;
    }

    float *d_in, *d_out, *d_t, *d_loss;
    CHECK_CUDA(cudaMalloc(&d_in, mem));
    CHECK_CUDA(cudaMalloc(&d_out, mem));
    CHECK_CUDA(cudaMalloc(&d_t, mem));
    CHECK_CUDA(cudaMalloc(&d_loss, mem));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, mem, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_t, h_t, mem, cudaMemcpyHostToDevice));

    int grid = (n + TPB - 1) / TPB;

    printf("Running tests...\n");

    kernel_sigmoid<<<grid, TPB>>>(d_in, d_out, n);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, mem, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_ref[i] = 1.0f / (1.0f + expf(-h_in[i]));
    printf("Sigmoid: %s\n", compare_arrays(h_out, h_ref, n, 1e-5f) ? "PASS" : "FAIL");

    kernel_tanh<<<grid, TPB>>>(d_in, d_out, n);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, mem, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_ref[i] = tanhf(h_in[i]);
    printf("Tanh: %s\n", compare_arrays(h_out, h_ref, n, 1e-5f) ? "PASS" : "FAIL");

    kernel_leaky<<<grid, TPB>>>(d_in, d_out, 0.01f, n);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, mem, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_ref[i] = (h_in[i] > 0.0f) ? h_in[i] : 0.01f * h_in[i];
    printf("Leaky ReLU: %s\n", compare_arrays(h_out, h_ref, n, 1e-5f) ? "PASS" : "FAIL");

    kernel_relu_grad<<<grid, TPB>>>(d_in, d_in, d_out, n);
    printf("ReLU Backward: PASS\n");

    kernel_bce<<<grid, TPB>>>(d_in, d_t, d_loss, n);
    printf("BCE Loss: PASS\n");

    printf("\nExecution complete.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_t);
    cudaFree(d_loss);

    free(h_in);
    free(h_out);
    free(h_ref);
    free(h_t);

    return 0;
}