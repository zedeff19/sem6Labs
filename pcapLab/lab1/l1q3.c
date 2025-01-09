#include "mpi.h"
#include <stdio.h>
#include <math.h> // For power and square root

int main(int argc, char *argv[]) {
    int rank, size;
    int a = 3, b = 12;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("My rank is %d in total %d processes.\n", rank, size);

    // Each process does a different operation depending on its rank
    if (rank == 0) {
        // Addition
        int ans = a + b;
        printf("a + b = %d\n", ans);
    }
    else if (rank == 1) {
        // Subtraction
        int ans = b - a;
        printf("b - a = %d\n", ans);
    }
    else if (rank == 2) {
        // Multiplication
        int ans = b * a;
        printf("b * a = %d\n", ans);
    }
    else if (rank == 3) {
        // Division
        if (a != 0) {
            int ans = b / a;
            printf("b / a = %d\n", ans);
        } else {
            printf("Division by zero is not allowed\n");
        }
    }
    else if (rank == 4) {
        // Moduluss
        int ans = b % a;
        printf("b %% a = %d\n", ans);
    }
    else if (rank == 5) {
        // Power (a^b)
        double ans = pow(a, b);
        printf("a ^ b = %.2f\n", ans);
    }
    else if (rank == 6) {
        // Square root of a
        if (a >= 0) {
            double ans = sqrt(a);
            printf("sqrt(a) = %.2f\n", ans);
        } else {
            printf("Cannot take square root of a negative number\n");
        }
    }
    else {
        printf("Rank %d does not have a defined operation\n", rank);
    }

    MPI_Finalize();

    return 0;
}


/*
My rank is 0 in total 4 processes.
a+b = 15.
My rank is 1 in total 4 processes.
b-a = 9.
My rank is 2 in total 4 processes.
b*a = 36.
My rank is 3 in total 4 processes.
b/a = 4.

*/
