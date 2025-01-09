#include "mpi.h"
#include <stdio.h>
#include <ctype.h> // For toupper() and tolower()
#include<string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char* str = "Hello World"; // Original string
    int len = strlen(str); // Length of the string

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure rank is less than the length of the string
    if (rank < len) {
        char local_char = str[rank]; // Get the character at index `rank`

        // Switch the case of the character
        if (isupper(local_char)) {
            local_char = tolower(local_char);
        } else if (islower(local_char)) {
            local_char = toupper(local_char);
        }

        // Print the modified character and the rank
        printf("Rank %d: Original char = '%c', Modified char = '%c'\n", rank, str[rank], local_char);
    } else {
        // Handle cases where there are more processes than characters
        printf("Rank %d: No character to modify (out of range).\n", rank);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
