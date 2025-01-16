#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, size;
    const int MAX_STRING_LENGTH = 100;
    char str[MAX_STRING_LENGTH];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if(rank == 0)
    {
        printf("Enter a string in the master process: ");
        scanf("%s", str);
        MPI_Send(str, strlen(str) + 1, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        fprintf(stdout, "I have sent %s from process 0\n", str);
        fflush(stdout);
    }
    else
    {
        MPI_Recv(str, MAX_STRING_LENGTH, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
        fprintf(stdout, "I have received %s in process %d\n", str, rank);
        size_t length = strlen(str);
        fprintf(stdout, "%zu is the length of the received string.\n", length);

        for(size_t i = 0 ; i < length; i++)
        {
            char local_char = str[i];
            if(isupper(local_char))
            {
                str[i] = tolower(local_char);
            }
            else if(islower(local_char))
            {
                str[i] = toupper(local_char);
            }
        }
        printf("Modified string: %s\n", str);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
