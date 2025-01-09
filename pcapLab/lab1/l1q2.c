#include "mpi.h"
#include<stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(rank%2==0)
    {
        printf("My rank is %d, I say Hello.\n", rank);
    }
    else{
        printf("My rank is %d, I say World.\n", rank);
    }


    MPI_Finalize();

    return 0;
}

/*
My rank is 0, I say Hello.
My rank is 1, I say World.
My rank is 2, I say Hello.
My rank is 3, I say World.
*/
