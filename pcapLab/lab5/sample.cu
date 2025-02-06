#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main(void)
{
    int a, b, c; // host cpoies of variables a,b,c
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    a=3;
    b=5;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    //launch add kerner on GPU
    add<<<1,1>>>(d_a,d_b,d_c); // 1, 1 means no of block, threads/block

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Result : %d", c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}