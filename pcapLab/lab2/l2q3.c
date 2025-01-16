#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if(rank == 0)
    {
        int array_size;
        printf("Enter the size of the array: ");
        scanf("%d", &array_size);
        int* array = (int*)malloc(array_size * sizeof(int)); // Allocate memory dynamically
        printf("Enter the array elements: ");
        for(int i = 0 ; i < array_size; i++)
        {
            scanf("%d", &array[i]);
        }

        for(int i = 1; i < size; i++)
        {
            if (i < array_size) {
                x = array[i];
                MPI_Send(&x, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
        }
        free(array);
        fprintf(stdout, "I have sent array elements from process 0 to all the slave processes.\n");
        fflush(stdout);
    }
    else
    {
        MPI_Recv(&x, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "I have received %d in process %d\n", x, rank);
        if(rank % 2 == 0)
        {
            printf("Rank: %d, Array element: %d, Square: %d\n", rank, x, x*x);
        }
        if(rank % 3 == 0)
        {
            printf("Rank: %d, Array element: %d, Cube: %d\n", rank, x, x*x*x);
        }
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
