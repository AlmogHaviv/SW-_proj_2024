#ifndef SYMNMF_H
#define SYMNMF_H

/* Declarations of the functions from symnmf.c */
double** allocate_matrix(int n, int k);
void free_matrix(double** matrix, int n);
double** sym(double** data, double** res, int n, int d);
double** ddg(double** data, double** res, int n, int d);
double** norm(double** data, double** res, int n, int d);
double** symnmf(int n, int k, double** W, double** H);

#endif