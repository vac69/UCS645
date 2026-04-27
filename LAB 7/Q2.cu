#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define SIZE 1000

double current_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void combine(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int leftArr[n1], rightArr[n2];

    for (int i = 0; i < n1; i++) leftArr[i] = arr[l + i];
    for (int j = 0; j < n2; j++) rightArr[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
    }

    while (i < n1) arr[k++] = leftArr[i++];
    while (j < n2) arr[k++] = rightArr[j++];
}

void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int mid = l + (r - l) / 2;
        merge_sort(arr, l, mid);
        merge_sort(arr, mid + 1, r);
        combine(arr, l, mid, r);
    }
}

int main() {
    int data[SIZE];
    int cpu_copy[SIZE];

    srand(time(NULL));

    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % 10000;
        cpu_copy[i] = data[i];
    }

    printf("Merge Sort Comparison (N = %d)\n\n", SIZE);

    double t1 = current_time();
    merge_sort(cpu_copy, 0, SIZE - 1);
    double cpu_duration = current_time() - t1;

    printf("CPU execution time: %.6f seconds\n", cpu_duration);

    thrust::device_vector<int> d_vec(data, data + SIZE);

    t1 = current_time();
    thrust::sort(d_vec.begin(), d_vec.end());
    cudaDeviceSynchronize();
    double gpu_duration = current_time() - t1;

    printf("GPU execution time: %.6f seconds\n", gpu_duration);
    printf("Speedup: %.2fx\n\n", cpu_duration / gpu_duration);

    printf("Execution completed successfully.\n");

    return 0;
}