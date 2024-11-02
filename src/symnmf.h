#ifndef SYMNMF_H
#define SYMNMF_H

/* Declarations of the functions from symnmf.c */
double** allocate_matrix(int n, int k);
void free_matrix(double** matrix, int n);
double** sym(double** data, int n, int d);
double** ddg(double** A, int n);
double** norm(double** A, double** D, int n);
double** symnmf(int n, int k, double** W, double** H);
void print_matrix(double** matrix, int n);

#endif