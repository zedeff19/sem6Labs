#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define matsize 3
#define mcw MPI_COMM_WORLD 

int main(int argc, char** argv) {
    int rank, size ;

    int matrix[matsize][matsize];
    int search_elem, loc_cnt = 0 , tot_cnt = 0;
    int row[matsize];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank == 0 )
    {
        printf("enter a 3*3 matrix: \n");
        for(int i = 0 ; i < matsize; i++)
        {
            for(int j = 0; j < matsize; j++)
            {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("enter element to be searched: ");
        scanf("%d", &search_elem);
    }

    MPI_Bcast(&search_elem, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        for(int i = 1; i < size; i++)
        {
            MPI_Send(matrix[i], matsize, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for(int j = 0; j < matsize; j++)
        {
            if(matrix[0][j]==search_elem)
                loc_cnt++;
        }
        printf("process %d, occurence of ele: %d \n", 0, loc_cnt);
    }    
    else{
        MPI_Recv(row, matsize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // row declared in the beginning, every process will have its own copy

        for(int j = 0 ; j < matsize; j++)
        {
            if(row[j] == search_elem)
                loc_cnt++;
        }

        printf("process %d, occurence of ele: %d \n", rank, loc_cnt);
    }
    
    MPI_Reduce(&loc_cnt, &tot_cnt, 1, MPI_INT, MPI_SUM, 0 , mcw);

    if(rank == 0 )
    {
        printf("total occurences of element %d in the matrix: %d\n", search_elem, tot_cnt);

    }

    MPI_Finalize();
    return 0;
}
