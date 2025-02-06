#include <cuda_runtime.h>
#include <stdio.h>


// CUDA kernel function to add two vectors
__global__ void addVectors(int *a, int *b, int *c, int arrSize) {
    int idx = blockIdx.x*256 + threadIdx.x; // Calculate thread index (part a)
    // int idx = blockIdx.x; // calc thread idx (part b)
    if (idx < arrSize) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n;
    printf("enter the array size: ");
    scanf("%d", &n);
    int numBlocks = 0 ;
    while(numBlocks * 256 < n)
    {
        numBlocks++;
    }

    // now numblocks is the min number of blocks with 256 threads each to calc  an array of size "n"
    printf("no of block to be allocated : %d\n", numBlocks);
    
    int a[n], b[n], c[n]; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    int size = n * sizeof(int);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
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
    addVectors<<<numBlocks, 256>>>(d_a, d_b, d_c, n);

    // Copy result vector from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

