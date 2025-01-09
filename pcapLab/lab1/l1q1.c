#include "mpi.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int x = 2;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double ans = pow(x, rank);

    printf("pow(%d, %d) = %.2f for rank: %d.\n", x, rank, ans, rank);

    MPI_Finalize();

    return 0;
}

/*
student@dbl-27:~/Documents/220962432_csaib73_pcap/lab1$ mpicc l1q1.c -o l1q1.out -lm
student@dbl-27:~/Documents/220962432_csaib73_pcap/lab1$ mpirun -np 4 ./l1q1.out
pow(2, 0) = 1.00 for rank: 0.
pow(2, 1) = 2.00 for rank: 1.
pow(2, 2) = 4.00 for rank: 2.
pow(2, 3) = 8.00 for rank: 3.
*/

