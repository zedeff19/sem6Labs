#include <cuda_runtime.h>
#include <stdio.h>

#define N 256 // Define the length of the vectors

// CUDA kernel function to add two vectors
__global__ void addVectors(int *a, int *b, int *c) {
    int idx = threadIdx.x; // Calculate thread index (part a)
    // int idx = blockIdx.x; // calc thread idx (part b)
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int a[N], b[N], c[N]; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    int size = N * sizeof(int);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy host vectors to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with N blocks and 1 thread per block
    addVectors<<<1, N>>>(d_a, d_b, d_c);

    // Copy result vector from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
