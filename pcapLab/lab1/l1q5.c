#include "mpi.h"
#include <stdio.h>

int fibo(int a) {
    if (a == 0) {
        return 0;
    } else if (a == 1) {
        return 1;
    } else {
        return fibo(a - 1) + fibo(a - 2);
    }
}

int factorial(int a) {
    if (a == 0 || a == 1) {
        return 1;
    } else {
        return a * factorial(a - 1);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("My rank is %d in total %d processes.\n", rank, size);

    if (rank % 2 == 0) {
        int fact = factorial(rank);
        printf("Rank %d is even, factorial: %d\n", rank, fact);
    }
    else {
        int fib = fibo(rank);
        printf("Rank %d is odd, Fibonacci: %d\n", rank, fib);
    }

    MPI_Finalize();

    return 0;
}
