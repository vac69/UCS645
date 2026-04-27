#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "CUBLAS %s:%d %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

#define BS 16

__global__ void matmulTile(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[BS][BS];
    __shared__ float sB[BS][BS];

    int r = blockIdx.y * BS + threadIdx.y;
    int c = blockIdx.x * BS + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + BS - 1) / BS; t++) {
        int a_col = t * BS + threadIdx.x;
        int b_row = t * BS + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (r < M && a_col < K) ? A[r*K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (c < N && b_row < K) ? B[b_row*N + c] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BS; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (r < M && c < N) {
        C[r*N + c] = acc;
    }
}

__global__ void pool2x2(const float *in, float *out, int n, int ch, int h, int w) {
    int h2 = h / 2;
    int w2 = w / 2;

    int bn = blockIdx.z;
    int bc = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= h2 || ow >= w2 || bn >= n || bc >= ch) return;

    float mx = -1e30f;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int ih = oh * 2 + i;
            int iw = ow * 2 + j;
            int idx = ((bn * ch + bc) * h + ih) * w + iw;
            mx = fmaxf(mx, in[idx]);
        }
    }

    out[((bn * ch + bc) * h2 + oh) * w2 + ow] = mx;
}

__global__ void bnInfer(const float *x, float *y, const float *g, const float *b,
                        const float *m, const float *v, int n, int ch, int hw, float eps) {
    int c = blockIdx.y;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= hw || c >= ch) return;

    for (int i = 0; i < n; i++) {
        int idx = (i * ch + c) * hw + id;
        float norm = (x[idx] - m[c]) / sqrtf(v[c] + eps);
        y[idx] = g[c] * norm + b[c];
    }
}

int main() {
    printf("\n========================================\n");
    printf(" CUDA CNN Kernels Test\n");
    printf("========================================\n");

    int M = 128, K = 128, N = 128;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);

    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 blk(BS, BS);
    dim3 grd((N + BS - 1) / BS, (M + BS - 1) / BS);

    matmulTile<<<grd, blk>>>(dA, dB, dC, M, N, K);
    printf("MatMul executed\n");

    int n = 4, c = 8, h = 16, w = 16;

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, (size_t)n*c*h*w*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (size_t)n*c*(h/2)*(w/2)*sizeof(float)));

    dim3 pblk(w/2, 2);
    dim3 pgrd((h/2 + 1)/2, c, n);

    pool2x2<<<pgrd, pblk>>>(dIn, dOut, n, c, h, w);
    printf("MaxPool executed\n");

    float *dx, *dy, *dg, *db, *dm, *dv;

    size_t bbytes = (size_t)n * c * h * w * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dx, bbytes));
    CHECK_CUDA(cudaMalloc(&dy, bbytes));
    CHECK_CUDA(cudaMalloc(&dg, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dm, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dv, c * sizeof(float)));

    dim3 bblk(256);
    dim3 bgrd((h*w + 255) / 256, c);

    bnInfer<<<bgrd, bblk>>>(dx, dy, dg, db, dm, dv, n, c, h*w, 1e-5f);

    printf("BatchNorm executed\n");

    printf("\nExecution finished.\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dIn); cudaFree(dOut);
    cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(db); cudaFree(dm); cudaFree(dv);

    free(hA); free(hB);

    return 0;
}