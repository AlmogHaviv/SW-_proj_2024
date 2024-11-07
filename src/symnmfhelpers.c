#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "symnmfhelpers.h"

#define BETA 0.5

/* Function to get dimensions (n and d) from the input file */
void get_dimensions(const char* filename, int* n, int* d) {
    FILE* file;
    char line[1024];
    *n = 0; /* Initialize number of rows */
    *d = 0; /* Initialize number of columns */

    file = fopen(filename, "r");
    if (!file) {
        printf("An Error Has Occurred\n");
        exit(EXIT_FAILURE);
    }

    /* Read the first line to determine the number of dimensions (d) */
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        while (token != NULL) {
            (*d)++;
            token = strtok(NULL, ",");
        }
        (*n)++; /* Count the first line */
    }

    /* Count the remaining lines to get the number of points (n) */
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
    }
    fclose(file);
}

/* Function to read data from input CSV file */
double** read_data(const char* filename, int* n, int* d) {
    FILE* file;
    double** data;
    char line[1024];
    int i = 0; int j;
    
    get_dimensions(filename, n, d);

    data = allocate_matrix(*n, *d);
    if (!data){
      printf("An Error Has Occurred\n");
      exit(1);
    }

    file = fopen(filename, "r");
    if (!file) {
        printf("An Error Has Occurred\n");
        free_matrix(data, *n);
        exit(1);
    }

    /* Rewind the file pointer to the beginning of the file */
    rewind(file);

    /* Read each line and fill the matrix */
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        j = 0;
        while (token != NULL) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;}

    fclose(file);
    return data;
}

double** allocate_matrix(int n, int k) {
    double** matrix;
    int i, j;

    matrix = (double**)calloc(n , sizeof(double*));
    if (!matrix) {
        return NULL;
    }

    for (i = 0; i < n; i++) {
        matrix[i] = (double*)calloc(k, sizeof(double));
        if (!matrix[i]) {
            /* Free previously allocated rows before exiting */
            for (j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

/* Function to free a matrix */
void free_matrix(double** matrix, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Function to compute the squared Euclidean distance */
double squared_euclidean(double* a, double* b, int d) {
    double sum = 0;
    int i;
    for (i = 0; i < d; i++) {
        sum += pow(a[i] - b[i], 2);
    }
    return sum;
}

/* Helper: Frobenius norm between two matrices */
double frobenius_norm(double** H, double** H_new, int n, int k) {
    double sum = 0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            sum += pow(H[i][j] - H_new[i][j], 2);
        }
    }
    return sum;
}

/* Function to multiply two matrices */
double** matrix_multiply(double** A, double** B, double** res, int m, int n, int p) {
    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            for (k = 0; k < n; k++) {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return res;
}

double** calc_transpose(double** matrix, int n, int k){
  int i, j;
  double** transposed = allocate_matrix(k, n);
  if (!transposed){
    return NULL;
  }
  for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
          transposed[i][j] = matrix[j][i]; /* Transpose */
      }
  }
  return transposed;
}

/* Function to update H matrix based on the update rule */
void update_H(double** H, double** W, int n, int k) {
    double** H_T;double** HHT;double** HHTH;
    int i, j;
    double denominator;
    double** WH = allocate_matrix(n, k);
    if(!WH){ 
      printf("An Error Has Occurred\n"); 
      free_matrix(H, n); free_matrix(W, n); exit(1);}
    matrix_multiply(W, H, WH, n, n, k);

    H_T = calc_transpose(H, n, k);
    if(!H_T){
      printf("An Error Has Occurred\n");
      free_matrix(H, n); free_matrix(W, n); free_matrix(WH, n); exit(1);
    }

    HHT = allocate_matrix(n, n);
    if (!HHT) {
        printf("An Error Has Occurred\n");
        free_matrix(H, n); free_matrix(W, n); free_matrix(WH, n); free_matrix(H_T, k); exit(1);
    }
    matrix_multiply(H, H_T, HHT, n, k, n);

    HHTH = allocate_matrix(n, k);
    if (!HHTH) {
        printf("An Error Has Occurred\n");
        free_matrix(H, n); free_matrix(W, n); free_matrix(WH, n); free_matrix(H_T, k); free_matrix(HHT, n); exit(1);
    }
    matrix_multiply(HHT, H, HHTH, n, n, k);

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            denominator = HHTH[i][j] != 0 ? HHTH[i][j] : 1;
            H[i][j] = H[i][j] * ((1 - BETA) + BETA * (WH[i][j] / denominator));
        }}

    /* Free temporary matrices */
    free_matrix(WH, n); free_matrix(H_T, k); free_matrix(HHT, n); free_matrix(HHTH, n);
}

/* Function to print a matrix */
void print_matrix(double** matrix, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < n - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}
