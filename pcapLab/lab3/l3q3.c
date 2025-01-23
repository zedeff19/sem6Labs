#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to count non-vowel characters in a string
int count_non_vowels(char *str, int length) {
    int count = 0;
    for (int i = 0; i < length; i++) {
        char c = tolower(str[i]);
        if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c >= 'a' && c <= 'z') {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    char *string = NULL;  // The input string
    char *local_string = NULL;  // Local portion of the string for each process
    int local_count = 0;  // Local count of non-vowels
    int *counts = NULL;  // Array to store counts from all processes
    int total_count = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads the string and N
        printf("Enter the number of processes N: ");
        fflush(stdout);
        scanf("%d", &N);

        if (size != N) {
            printf("Error: The number of processes must match N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read the string from the user
        printf("Enter the string: ");
        fflush(stdout);
        string = (char*)malloc(1000 * sizeof(char));  // Allocate memory for string
        scanf("%s", string);

        int string_length = strlen(string);
        if (string_length % N != 0) {
            printf("Error: The length of the string must be divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory to store counts from all processes
        counts = (int*)malloc(N * sizeof(int));

        // Allocate memory for scattering: split string into N parts
        int portion_size = string_length / N;
        local_string = (char*)malloc((portion_size + 1) * sizeof(char));

        // Scatter the string to processes (send portions of the string)
        MPI_Scatter(string, portion_size, MPI_CHAR, local_string, portion_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Null terminate the string for each process (important for safety)
        local_string[portion_size] = '\0';

        // Count non-vowels in the local portion
        local_count = count_non_vowels(local_string, portion_size);

        // Gather the counts from all processes
        MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Root process calculates the total count of non-vowels and prints the results
        printf("Non-vowels found by each process:\n");
        total_count = 0;
        for (int i = 0; i < N; i++) {
            printf("Process %d: %d non-vowels\n", i, counts[i]);
            total_count += counts[i];
        }
        printf("Total number of non-vowels: %d\n", total_count);

        // Clean up memory
        free(string);
        free(counts);
    } else {
        // Non-root processes receive their portion of the string
        int portion_size;
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, &portion_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        local_string = (char*)malloc((portion_size + 1) * sizeof(char));  // Allocate memory for the local string
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, local_string, portion_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Null terminate the string for each process (important for safety)
        local_string[portion_size] = '\0';

        // Count non-vowels in the local portion
        local_count = count_non_vowels(local_string, portion_size);

        // Send the local count back to root
        MPI_Gather(&local_count, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

        // Clean up memory
        free(local_string);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
