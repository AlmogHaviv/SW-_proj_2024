#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>

/* Convert a Python list of lists (py_matrix) to a C matrix */
static double** python_matrix_to_c_matrix(PyObject* py_matrix, int n, int d) {
    double** c_matrix = (double**)calloc(n, sizeof(double*));
    if (!c_matrix) return NULL;  

    for (int i = 0; i < n; i++) {
        c_matrix[i] = (double*)calloc(d, sizeof(double));
        if (!c_matrix[i]) return NULL;  

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

/* Helper function to get the number of rows and cols */
void calc_dimensions(PyObject *py_matrix, int *rows, int *cols) {
    *rows = PyList_Size(py_matrix);
    if (*rows == 0) {
        printf("An Error Has Occurred\n");
        return;
    }
    *cols = PyList_Size(PyList_GetItem(py_matrix, 0));
}

/* API function sym */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject *data_array;
    int n = 0;
    int d = 0;
    if (!PyArg_ParseTuple(args, "O", &data_array)) return NULL;

    calc_dimensions(data_array, &n, &d);
    double** data = python_matrix_to_c_matrix(data_array, n, d);
    double** res = allocate_matrix(n, n);
    if(!res){
        printf("An Error Has Occurred\n");
        free_matrix(data, n);
        return NULL;  
    }
    sym(data, res, n, d);

    PyObject* result_array = c_matrix_to_python_matrix(res, n, n);
    free_matrix(data, n); free_matrix(res, n);

    return result_array;
}

/* API function ddg */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject *data_array;
    int n = 0;
    int d = 0;
    if (!PyArg_ParseTuple(args, "O", &data_array)) return NULL;

    calc_dimensions(data_array, &n, &d);
    double** data = python_matrix_to_c_matrix(data_array, n, d);
    double** res = allocate_matrix(n, n);
    if(!res){
        printf("An Error Has Occurred\n");
        free_matrix(data, n);
        return NULL;  
    }
    ddg(data, res, n, d);

    PyObject* result_array = c_matrix_to_python_matrix(res, n, n);
    free_matrix(data, n); free_matrix(res, n);

    return result_array;
}

/* API function norm */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject *data_array;
    int n = 0;
    int d = 0;
    if (!PyArg_ParseTuple(args, "O", &data_array)) return NULL;

    calc_dimensions(data_array, &n, &d);
    double** data = python_matrix_to_c_matrix(data_array, n, d);
    double** res = allocate_matrix(n, n);
    if(!res){
        printf("An Error Has Occurred\n");
        free_matrix(data, n);
        return NULL;  
    }
    norm(data, res, n, d);

    PyObject* result_array = c_matrix_to_python_matrix(res, n, n);
    free_matrix(data, n); free_matrix(res, n);

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

    free_matrix(W, n); free_matrix(H, k);
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
