#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"  /* Include the header file */
#include <stdio.h>



/* Helper functions to convert Python list to C matrix and vice versa */
double** py_list_to_c_matrix(PyObject* py_list, int n, int d);
PyObject* c_matrix_to_py_list(double** matrix, int n, int k);
void print_matrix1(double** matrix, int n, int d);

/* Function to print a matrix of size n rows and d columns */
void print_matrix1(double** matrix, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            /* Print each element with a space in between and 4 decimal precision */
            printf("%.4f ", matrix[i][j]);
        }
        /* Newline after each row */
        printf("\n");
    }
}

/* Helper function to check dimensions of input list */
static int validate_dimensions(PyObject* py_list, int expected_n, int expected_d) {
    if (!PyList_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return 0;
    }

    Py_ssize_t actual_n = PyList_Size(py_list);
    if (actual_n != expected_n) {
        PyErr_SetString(PyExc_ValueError, "Mismatched number of rows");
        return 0;
    }

    for (Py_ssize_t i = 0; i < actual_n; i++) {
        PyObject* py_row = PyList_GetItem(py_list, i);
        if (!PyList_Check(py_row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
            return 0;
        }

        Py_ssize_t actual_d = PyList_Size(py_row);
        if (actual_d != expected_d) {
            PyErr_SetString(PyExc_ValueError, "Mismatched row length");
            return 0;
        }
    }
    return 1;
}

double** py_list_to_c_matrix(PyObject* py_list, int n, int d) {
    /* Validate input dimensions first */
    if (!validate_dimensions(py_list, n, d)) {
        return NULL;
    }

    /* Allocate memory for the C matrix */
    double** matrix = allocate_matrix(n, d);
    if (!matrix) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    /* Iterate over the Python list and convert it to a C matrix */
    for (int i = 0; i < n; i++) {
        PyObject* py_row = PyList_GetItem(py_list, i);

        for (int j = 0; j < d; j++) {
            PyObject* py_item = PyList_GetItem(py_row, j);

            /* Handle both float and integer inputs */
            if (PyFloat_Check(py_item)) {
                matrix[i][j] = PyFloat_AsDouble(py_item);
            } else if (PyLong_Check(py_item)) {
                matrix[i][j] = (double)PyLong_AsLong(py_item);
            } else {
                PyErr_SetString(PyExc_TypeError, "Matrix elements must be numbers");
                free_matrix(matrix, n);
                return NULL;
            }

            if (PyErr_Occurred()) {
                free_matrix(matrix, n);
                return NULL;
            }
        }
    }

    return matrix;
}


/* Helper function to convert C matrix to Python list */
PyObject* c_matrix_to_py_list(double** matrix, int n, int k) {
    /* Create a Python list to hold the result */
    PyObject* py_list = PyList_New(n);

    /* Fill the Python list with the values from the C matrix */
    for (int i = 0; i < n; i++) {
        PyObject* py_row = PyList_New(k);
        for (int j = 0; j < k; j++) {
            PyObject* py_value = PyFloat_FromDouble(matrix[i][j]);  /* Convert C double to Python float */
            PyList_SetItem(py_row, j, py_value);  /* Set the value in the Python row */
        }
        PyList_SetItem(py_list, i, py_row);  /* Set the row in the Python list */
    }

    return py_list;
}


/* Modified wrapper function */
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_data;
    int n, d;

    if (!PyArg_ParseTuple(args, "Oii", &py_data, &n, &d)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }

    /* Convert Python list to C matrix */
    double** data = py_list_to_c_matrix(py_data, n, d);
    if (!data) {
        return NULL;  // Error already set by py_list_to_c_matrix
    }

    /* Call the sym() function from symnmf_functions.c */
    double** result = sym(data, n, d);
    if (!result) {
        free_matrix(data, n);
        PyErr_SetString(PyExc_RuntimeError, "sym() function failed");
        return NULL;
    }

    /* Convert result to Python list */
    PyObject* py_result = c_matrix_to_py_list(result, n, n);

    /* Free allocated memory */
    free_matrix(result, n);
    free_matrix(data, n);

    if (!py_result) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert result to Python list");
        return NULL;
    }

    return py_result;
}


/* Wrapper function for ddg() */
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_data;
    int n, d;

    if (!PyArg_ParseTuple(args, "Oii", &py_data, &n, &d)) {
        return NULL;  /* Invalid arguments */
    }

    /* Convert Python list to C matrix */
    double** A = py_list_to_c_matrix(py_data, n, d);

    /* Call the ddg() function from symnmf_functions.c */
    double** D = ddg(A, n, d);

    /* Convert result to Python list */
    PyObject* py_result = c_matrix_to_py_list(D, n, n);

    /* Free allocated memory */
    free_matrix(D, n);
    free_matrix(A, n);

    return py_result;
}

/* Wrapper function for norm() */
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_A, *py_D;
    int n;

    if (!PyArg_ParseTuple(args, "OOi", &py_A, &py_D, &n)) {
        return NULL;  /* Invalid arguments */
    }

    /* Convert Python lists to C matrices */
    double** A = py_list_to_c_matrix(py_A, n, n);
    double** D = py_list_to_c_matrix(py_D, n, n);

    /* Call the norm() function from symnmf_functions.c */
    double** W = norm(A, D, n);

    /* Convert result to Python list */
    PyObject* py_result = c_matrix_to_py_list(W, n, n);

    /* Free allocated memory */
    free_matrix(W, n);
    free_matrix(A, n);
    free_matrix(D, n);

    return py_result;
}

/* Wrapper function for symnmf() */
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_W, *py_H;
    int n, k;

    if (!PyArg_ParseTuple(args, "iOOi", &n, &py_W, &py_H, &k)) {
        return NULL;  /* Invalid arguments */
    }

    /* Convert Python lists to C matrices */
    double** W = py_list_to_c_matrix(py_W, n, n);
    double** H = py_list_to_c_matrix(py_H, n, k);

    /* Call the symnmf() function from symnmf_functions.c */
    double** H_result = symnmf(n, k, W, H);

    /* Convert result to Python list */
    PyObject* py_result = c_matrix_to_py_list(H_result, n, k);

    /* Free allocated memory */
    free_matrix(H_result, n);
    free_matrix(W, n);
    free_matrix(H, n);

    return py_result;
}

/* Method definition table */
static PyMethodDef SymNMFMethods[] = {
    {"sym", sym_wrapper, METH_VARARGS, "Compute the similarity matrix"},
    {"ddg", ddg_wrapper, METH_VARARGS, "Compute the diagonal degree matrix"},
    {"norm", norm_wrapper, METH_VARARGS, "Compute the normalized similarity matrix"},
    {"symnmf", symnmf_wrapper, METH_VARARGS, "Perform Symmetric NMF"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",  /* Module name */
    NULL,      /* Documentation (optional) */
    -1,        /* Size of per-interpreter state of the module */
    SymNMFMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
