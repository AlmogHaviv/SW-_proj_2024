#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "symnmfhelpers.h"

#define MAX_ITER 300
#define EPSILON 1e-4

/* Function Declarations */
double** sym(double** data, double** res, int n, int d);
double** ddg(double** data, double** res, int n, int d);
double** norm(double** data, double** res, int n, int d);
double** symnmf(int n, int k, double** W, double** H);

int main(int argc, char* argv[]) {
    char* goal;
    char* file_name;
    int n, d;
    double** data; double** res_matrix;

    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];
    data = read_data(file_name, &n, &d);
    res_matrix = allocate_matrix(n, n);
    if(!res_matrix){
       printf("An Error Has Occurred\n");
       return 1; 
    }

    if (strcmp(goal, "sym") == 0) {
        sym(data, res_matrix, n, d);
        
    } else if (strcmp(goal, "ddg") == 0) {
        ddg(data, res_matrix, n, d);

    } else if (strcmp(goal, "norm") == 0) {
        norm(data, res_matrix, n, d);

    } else {
        printf("An Error Has Occurred\n");
        free_matrix(data, n); free_matrix(res_matrix, n);
        return 1;
    }
    print_matrix(res_matrix, n);
    free_matrix(data, n); free_matrix(res_matrix, n);
    return 0;
}

/* 1.1: Calculate the Similarity Matrix */
double** sym(double** data, double** res, int n, int d) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                res[i][j] = 0;
            } else {
                res[i][j] = exp(-squared_euclidean(data[i], data[j], d) / 2);
            }
        }
    }
    return res;
}

/* 1.2: Calculate the Diagonal Degree Matrix */
double** ddg(double** data, double** res, int n, int d) {
    int i, j;
    double sum;
    double** sym_matrix = allocate_matrix(n, n);
    if(!sym_matrix){
        free_matrix(data, n);
        free_matrix(res, n);
        printf("An Error Has Occurred\n");
        exit(1); 
    }
    sym(data, sym_matrix, n, d);
    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < n; j++) {
            sum += sym_matrix[i][j];
            res[i][j] = 0.0;
        }
        res[i][i] = sum;
    }
    free_matrix(sym_matrix, n);
    return res;
}

/* 1.3: Calculate the Normalized Similarity Matrix */
double** norm(double** data, double** res, int n, int d) {
    double** sym_matrix; double** ddg_matrix; double** ddg_sqrt; double** temp;
    int i; 
    double value, sqrt_value;
    sym_matrix = allocate_matrix(n, n);
    if (!sym_matrix){
        free_matrix(data, n); free_matrix(res, n);
        printf("An Error Has Occurred\n"); exit(1); }
    sym(data, sym_matrix, n, d);

    ddg_matrix = allocate_matrix(n, n);
    if (!ddg_matrix){
        free_matrix(data, n); free_matrix(res, n); free_matrix(sym_matrix, n);
        printf("An Error Has Occurred\n"); exit(1); }
    ddg(data, ddg_matrix, n, d);

    ddg_sqrt = allocate_matrix(n, n);
    if (!ddg_sqrt){
        free_matrix(data, n); free_matrix(res, n); free_matrix(sym_matrix, n); free_matrix(ddg_matrix, n);
        printf("An Error Has Occurred\n"); exit(1); }

    temp = allocate_matrix(n, n);
    if (!temp){
        free_matrix(data, n); free_matrix(res, n); free_matrix(sym_matrix, n); free_matrix(ddg_matrix, n); free_matrix(ddg_sqrt, n);
        printf("An Error Has Occurred\n"); exit(1); }
    /* Calculate ddg^(-1/2) */
    for (i = 0; i < n; i++) {
        value = ddg_matrix[i][i];
        sqrt_value = (value > 0.0) ? 1.0 / sqrt(value) : 0.0;
        ddg_sqrt[i][i] = sqrt_value;
    }
    /* Calculate res = ddg_sqrt * sym_matrix * ddg_sqrt */
    matrix_multiply(ddg_sqrt, sym_matrix, temp, n, n, n); /* ddg_sqrt * sym_matrix */
    matrix_multiply(temp, ddg_sqrt, res, n, n, n); /* (ddg_sqrt * sym_matrix) * ddg_sqrt */
    free_matrix(temp, n); free_matrix(sym_matrix, n); free_matrix(ddg_sqrt, n); free_matrix(ddg_matrix, n);
    return res;
}

/* Symmetric NMF algorithm */
double** symnmf(int n, int k, double** W, double** H) {
    int iter = 0;
    int i, j;
    double norm;
    double** temp = allocate_matrix(n, k);
    if(!temp){
       free_matrix(W, n); free_matrix(W, n);
       printf("An Error Has Occurred\n"); exit(1);  
    }

    while (iter < MAX_ITER) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                temp[i][j] = H[i][j]; /* Transpose */
            }
        }

        update_H(H, W, n, k);
        norm = frobenius_norm(temp, H, n, k);
        if (norm < EPSILON) break;
        iter++;
    }
    free_matrix(temp, n);
    return H;  
}
