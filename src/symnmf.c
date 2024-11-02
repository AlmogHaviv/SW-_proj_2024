#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5

/* Function Declarations */
double** read_data(const char* filename, int* n, int* d);
void get_dimensions(const char* filename, int* n, int* d);
double** allocate_matrix(int n, int k);
void free_matrix(double** matrix, int n);
double squared_euclidean(double* a, double* b, int d);
double** sym(double** data, int n, int d);
double** ddg(double** A, int n);
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

    if (argc != 3) {
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
        double** D = ddg(A, n);
        print_matrix(D, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        double** A = sym(data, n, d);
        double** D = ddg(A, n);
        double** W = norm(A, D, n);
        print_matrix(W, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free_matrix(W, n);
    }

    free_matrix(data, n);
    return 0;
}

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
    int i = 0; /* Initialize row index */
    int j; /* Column index */

    /* Get dimensions (n and d) from the file */
    get_dimensions(filename, n, d);

    /* Allocate memory for the data matrix */
    data = allocate_matrix(*n, *d);

    file = fopen(filename, "r");
    if (!file) {
        printf("An Error Has Occurred\n");
        exit(EXIT_FAILURE);
    }

    /* Rewind the file pointer to the beginning of the file */
    rewind(file);

    /* Read each line and fill the matrix */
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        j = 0; /* Initialize column index */
        while (token != NULL) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
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
double** ddg(double** A, int n) {
    double** D = allocate_matrix(n, n);
    int i, j;

    for (i = 0; i < n; i++) {
        double sum = 0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
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
    double** temp = allocate_matrix(n, n);
    int i;
    double threshold = 1e-10;

    /* Initialize D^(-1/2) with zeros and set the diagonal elements */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            D_sqrt[i][j] = 0.0;  // Set all elements to zero initially
        }
    }


    /* Calculate D^(-1/2) */
    for (i = 0; i < n; i++) {
       double value = D[i][i];
        double sqrt_value = (value > threshold) ? 1.0 / sqrt(value) : 1.0 / sqrt(threshold);
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
    if (C == NULL) {
        return NULL;
    }

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
    if (WH == NULL) {
        return;
    }

    /* Calculate transpose H^T */
    double** H_T = allocate_matrix(k, n);
    if (H_T == NULL) {
        free_matrix(WH, n);
        return;
    }

    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
            H_T[i][j] = H[j][i]; /* Transpose */
        }
    }

    /* Calculate H(H^T)H */
    double** HHT = matrix_multiply(H, H_T, n, k, n);
    if (HHT == NULL) {
        free_matrix(WH, n);
        free_matrix(H_T, k);
        return;
    }

    double** HHTH = matrix_multiply(HHT, H, n, n, k);
    if (HHTH == NULL) {
        free_matrix(WH, n);
        free_matrix(H_T, k);
        free_matrix(HHT, n);
        return;
    }

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

/* Symmetric NMF algorithm */
double** symnmf(int n, int k, double** W, double** H) {
    int iter = 0;
    double** temp = allocate_matrix(n, k);
    int i, j;


    while (iter < MAX_ITER) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                temp[i][j] = H[i][j]; /* Transpose */
            }
        }

        update_H(H, W, n, k);

        double norm = frobenius_norm(temp, H, n, k);

        if (norm < EPSILON) break;
        iter++;
    }
    free_matrix(temp, k);

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
