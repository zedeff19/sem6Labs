#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

void check_mpi_error(int error_code) {
    if (error_code != MPI_SUCCESS) {
        char error_string[256];
        int length_of_error_string;
        MPI_Error_string(error_code, error_string, &length_of_error_string);
        fprintf(stderr, "MPI error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, error_code);  // Abort the program
    }
}

int main(int argc, char** argv) {
    int rank, size, fact = 1, res = 0, error_code;

    error_code = MPI_Init(&argc, &argv);
    check_mpi_error(error_code);
    
    error_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    check_mpi_error(error_code);

    error_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    check_mpi_error(error_code);
    
    for (int i = 1; i <= rank + 1; i++) {
        fact *= i;
    }

    error_code = MPI_Scan(&fact, &res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    check_mpi_error(error_code);

    printf("Process %d: Scan result = %d\n", rank, res);

    error_code = MPI_Finalize();
    check_mpi_error(error_code);

    return 0;
}
