#ifndef SYMNMFHELPERS_H
#define SYMNMFHELPERS_H

/* Declarations of the functions from symnmfhelpers.c */
double** read_data(const char* filename, int* n, int* d);

void get_dimensions(const char* filename, int* n, int* d);

double** allocate_matrix(int n, int k);

void free_matrix(double** matrix, int n);

double squared_euclidean(double* a, double* b, int d);

double frobenius_norm(double** H, double** H_new, int n, int k);

double** matrix_multiply(double** A, double** B, double** res, int m, int n, int p);

double** calc_transpose(double** matrix, int n, int k);

void update_H(double** H, double** W, int n, int k);

void print_matrix(double** matrix, int n);

#endif