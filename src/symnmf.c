#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5

/* Function Declarations */
double** read_data(const char* filename, int* n, int* d);
double** allocate_matrix(int n, int k);
void free_matrix(double** matrix, int n);
double squared_euclidean(double* a, double* b, int d);
double** sym(double** data, int n, int d);
double** ddg(double** A, int n, int d);
double** norm(double** A, double** D, int n);
double frobenius_norm(double** H, double** H_new, int n, int k);
double** matrix_multiply(double** A, double** B, int m, int n, int p);
void update_H(double** H, double** W, int n, int k);
double** symnmf(int n, int k, double** W, double** H);
void print_matrix(double** matrix, int n);

int main(int argc, char* argv[]) {
    char* goal;
    char* file_name;
    int n, d;
    double** data;

    if (argc < 3) {
        printf("An Error Has Occurred");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];
    data = read_data(file_name, &n, &d);

    if (strcmp(goal, "sym") == 0) {
        double** A = sym(data, n, d);
        print_matrix(A, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        double** A = sym(data, n, d);
        double** D = ddg(A, n, d);
        print_matrix(D, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        double** A = sym(data, n, d);
        double** D = ddg(A, n, d);
        double** W = norm(A, D, n);
        print_matrix(W, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free_matrix(W, n);
    }

    free_matrix(data, n);
    return 0;
}

/* Function to read data from input file */
double** read_data(const char* filename, int* n, int* d) {
    FILE* file;
    double** data;
    int i, j;

    file = fopen(filename, "r");
    if (!file) {
        printf("An Error Has Occurred");
        exit(EXIT_FAILURE);
    }

    /* First, read the number of points (n) and dimensions (d) */
    if (fscanf(file, "%d %d", n, d) != 2) {
        printf("An Error Has Occurred\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the data matrix */
    data = allocate_matrix(*n, *d);

    /* Read the data points */
    for (i = 0; i < *n; i++) {
        for (j = 0; j < *d; j++) {
            if (fscanf(file, "%lf", &data[i][j]) != 1) {
                printf("An Error Has Occurred\n");
                fclose(file);
                free_matrix(data, *n);  // Free allocated memory before exiting
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
    return data;
}

double** allocate_matrix(int n, int k) {
    double** matrix;
    int i, j;

    matrix = (double**) malloc(n * sizeof(double*));
    if (!matrix) {
        printf("An Error Has Occurred\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < n; i++) {
        matrix[i] = (double*) malloc(k * sizeof(double));
        if (!matrix[i]) {
            /* Free previously allocated rows before exiting */
            for (j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            printf("An Error Has Occurred\n");
            exit(EXIT_FAILURE);
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

/* 1.1: Calculate the Similarity Matrix */
double** sym(double** data, int n, int d) {
    double** A;
    int i, j;

    A = allocate_matrix(n, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0;
            } else {
                A[i][j] = exp(-squared_euclidean(data[i], data[j], d) / 2);
            }
        }
    }
    return A;
}

/* 1.2: Calculate the Diagonal Degree Matrix */
double** ddg(double** A, int n, int d) {
    double** D = allocate_matrix(n, n);
    int i, j;

    for (i = 0; i < n; i++) {
        double sum = 0;
        for (j = 0; j < n; j++) {
            if (j < d){
                sum += A[i][j];
            }
            D[i][j] = 0.0;
        }
        D[i][i] = sum;
    }
    return D;
}

/* 1.3: Calculate the Normalized Similarity Matrix */
double** norm(double** A, double** D, int n) {
    double** W = allocate_matrix(n, n);
    double** D_sqrt = allocate_matrix(n, n);
    double** temp;
    int i;

    /* Calculate D^(-1/2) */
    for (i = 0; i < n; i++) {
        double sqrt_value = (D[i][i] != 0) ? 1.0 / sqrt(D[i][i]) : 0.0;  /* Avoid division by zero */
        D_sqrt[i][i] = sqrt_value;
    }

    /* Calculate W = D_sqrt * A * D_sqrt */
    temp = matrix_multiply(D_sqrt, A, n, n, n); /* D_sqrt * A */
    W = matrix_multiply(temp, D_sqrt, n, n, n); /* (D_sqrt * A) * D_sqrt */

    /* Free temporary matrix */
    free_matrix(temp, n);
    free_matrix(D_sqrt, n);

    return W;
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
double** matrix_multiply(double** A, double** B, int m, int n, int p) {
    double** C = allocate_matrix(m, p);
    int i, j, k;

    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0;
            for (k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

/* Function to update H matrix based on the update rule */
void update_H(double** H, double** W, int n, int k) {
    /* Calculate WH */
    double** WH = matrix_multiply(W, H, n, n, k);
    double** H_T = allocate_matrix(k, n);
    int i, j;
    double** HHT;
    double** HHTH;

    /* Calculate H^T */
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
            H_T[i][j] = H[j][i]; /* Transpose */
        }
    }

    /* Calculate H(H^T)H */
    HHT = matrix_multiply(H, H_T, n, k, n);
    HHTH = matrix_multiply(HHT, H, n, n, k);

    /* Update H */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double denominator = HHTH[i][j] != 0 ? HHTH[i][j] : 1;
            H[i][j] = H[i][j] * ((1 - BETA) + BETA * (WH[i][j] / denominator));
        }
    }

    /* Free temporary matrices */
    free_matrix(WH, n);
    free_matrix(H_T, k);
    free_matrix(HHT, n);
    free_matrix(HHTH, n);
}

/* 1.4: Symmetric NMF algorithm */
double** symnmf(int n, int k, double** W, double** H) {
    double** H_new = allocate_matrix(n, k);
    int iter = 0;
    double** temp;

    while (iter < MAX_ITER) {
        update_H(H_new, W, n, k);
        if (frobenius_norm(H, H_new, n, k) < EPSILON) break;

        temp = H;
        H = H_new;
        H_new = temp;
        iter++;
    }

    free_matrix(H_new, n);
    return H;  /* Memory to be freed externally */
}

/* Function to print a matrix */
void print_matrix(double** matrix, int n) {
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < n - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
}
