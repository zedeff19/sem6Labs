// my aim is to add two arrays of 20 elements eeach such that there 
// are 2 blocks with 10 threads each

#include <cuda_runtime.h>
#include <stdio.h>

#define N 20

__global__ void addVector(int *a, int *b, int *c)
{
    int idx = blockIdx.x*10 + threadIdx.x;
    if(idx<20)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{

    int a[N], b[N], c[N]; // these are my local(host) elements
    int *d_a, *d_b, *d_c; // these will be allocated mem in thge device i.e. GPU

    int size = N*sizeof(int);

    for(int i = 0 ; i < N ; i++) //initializ the input arrays
    {
        a[i] = i;
        b[i] = 2*i;
    }

    cudaMalloc((void **)&d_a, size); // allocating memories to d_xs in the gpu memory
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    addVector<<<2,10>>>(d_a, d_b, d_c); // launching 2 blocks with 10 threads each

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

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