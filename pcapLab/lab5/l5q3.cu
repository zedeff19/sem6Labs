#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 20

__global__ void convert(int *a, float *b)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        // Convert degrees to radians and then compute sine
        b[idx] = sinf(a[idx] * M_PI / 180.0f);
    }
}

int main()
{
    int a[N];
    float b[N];
    int *d_a;
    float *d_b;

    for (int i = 0; i < N; i++)
    {
        a[i] = 10 * i; // Input values in degrees
    }

    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    convert<<<1, N>>>(d_a, d_b);

    cudaMemcpy(b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
