#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>

/* Helper function declarations */
static double** python_matrix_to_c_matrix(PyObject* py_matrix, int n, int d);
static PyObject* c_matrix_to_python_matrix(double** matrix, int n, int d);

/* Convert a Python list of lists (py_matrix) to a C matrix */
static double** python_matrix_to_c_matrix(PyObject* py_matrix, int n, int d) {
    double** c_matrix = (double**)malloc(n * sizeof(double*));
    if (!c_matrix) return NULL;  /* Check allocation */

    for (int i = 0; i < n; i++) {
        c_matrix[i] = (double*)malloc(d * sizeof(double));
        if (!c_matrix[i]) return NULL;  /* Check allocation */

        PyObject* row = PyList_GetItem(py_matrix, i);
        for (int j = 0; j < d; j++) {
            PyObject* item = PyList_GetItem(row, j);
            c_matrix[i][j] = PyFloat_AsDouble(item);
        }
    }
    return c_matrix;
}

/* Convert a C matrix to a Python list of lists */
static PyObject* c_matrix_to_python_matrix(double** matrix, int n, int d) {
    PyObject* py_matrix = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyObject* item = PyFloat_FromDouble(matrix[i][j]);
            PyList_SetItem(row, j, item);
        }
        PyList_SetItem(py_matrix, i, row);
    }
    return py_matrix;
}

/* Helper function to free a dynamically allocated C matrix */
static void free_c_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* API function sym */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject *data_array;
    int n, d;
    if (!PyArg_ParseTuple(args, "Oii", &data_array, &n, &d)) return NULL;

    double** data = python_matrix_to_c_matrix(data_array, n, d);
    double** A = sym(data, n, d);

    PyObject* result_array = c_matrix_to_python_matrix(A, n, n);

    free_c_matrix(data, n);
    free_c_matrix(A, n);

    return result_array;
}

/* API function ddg */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject *sym_array;
    int n;
    if (!PyArg_ParseTuple(args, "Oi", &sym_array, &n)) return NULL;

    double** sym_matrix = python_matrix_to_c_matrix(sym_array, n, n);
    double** D = ddg(sym_matrix, n);

    PyObject* result_array = c_matrix_to_python_matrix(D, n, n);

    free_c_matrix(sym_matrix, n);
    free_c_matrix(D, n);

    return result_array;
}

/* API function norm */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject *sym_array, *ddg_array;
    int n;
    if (!PyArg_ParseTuple(args, "OOi", &sym_array, &ddg_array, &n)) return NULL;

    double** sym_matrix = python_matrix_to_c_matrix(sym_array, n, n);
    double** ddg_matrix = python_matrix_to_c_matrix(ddg_array, n, n);
    double** W = norm(sym_matrix, ddg_matrix, n);

    PyObject* result_array = c_matrix_to_python_matrix(W, n, n);

    free_c_matrix(sym_matrix, n);
    free_c_matrix(ddg_matrix, n);
    free_c_matrix(W, n);

    return result_array;
}

/* API function symnmf */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    int n, k;
    PyObject *W_array, *H_array;
    if (!PyArg_ParseTuple(args, "iiOO", &n, &k, &W_array, &H_array)) return NULL;

    double** W = python_matrix_to_c_matrix(W_array, n, n);
    double** H = python_matrix_to_c_matrix(H_array, n, k);
    double** H_final = symnmf(n, k, W, H);

    PyObject* result_array = c_matrix_to_python_matrix(H_final, n, k);

    free_c_matrix(W, n);
    free_c_matrix(H, k);

    return result_array;
}

/* Module methods */
static PyMethodDef SymnmfMethods[] = {
    {"sym", sym_wrapper, METH_VARARGS, "Calculate the similarity matrix."},
    {"ddg", ddg_wrapper, METH_VARARGS, "Calculate the diagonal degree matrix."},
    {"norm", norm_wrapper, METH_VARARGS, "Calculate the normalized similarity matrix."},
    {"symnmf", symnmf_wrapper, METH_VARARGS, "Perform symmetric nmf."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
