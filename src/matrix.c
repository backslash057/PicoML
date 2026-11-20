#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Matrix* matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        fprintf(stderr, "[Error] matrix_create: Dimensions must be > 0 (got %zu x %zu)\n", rows, cols);
        return NULL;
    }

    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "[Error] matrix_create: Failed to allocate Matrix struct\n");
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)malloc(rows * cols * sizeof(float));
    if (!mat->data) {
        fprintf(stderr, "[Error] matrix_create: Failed to allocate matrix buffer\n");
        free(mat);
        return NULL;
    }

    return mat;
}

Matrix* matrix_zeros(size_t rows, size_t cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    memset(mat->data, 0, rows * cols * sizeof(float));
    return mat;
}

Matrix* matrix_ones(size_t rows, size_t cols) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        mat->data[i] = 1.0f;
    }
    return mat;
}

Matrix* matrix_randn(size_t rows, size_t cols, float mean, float std) {
    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        /* Box-Muller transform */
        float u1 = (float)rand() / (float)RAND_MAX;
        while (u1 <= 1e-7f) {
            u1 = (float)rand() / (float)RAND_MAX;
        }
        float u2 = (float)rand() / (float)RAND_MAX;

        float z = sqrtf(-2.0f * logf(u1)) * cosf((float)(2.0 * M_PI * u2));
        mat->data[i] = mean + std * z;
    }

    return mat;
}

Matrix* matrix_from_array(const float* arr, size_t rows, size_t cols) {
    if (!arr) {
        fprintf(stderr, "[Error] matrix_from_array: Source array is NULL\n");
        return NULL;
    }

    Matrix* mat = matrix_create(rows, cols);
    if (!mat) return NULL;

    memcpy(mat->data, arr, rows * cols * sizeof(float));
    return mat;
}

void matrix_free(Matrix* mat) {
    if (!mat) return;
    if (mat->data) {
        free(mat->data);
        mat->data = NULL;
    }
    free(mat);
}

void matrix_print(const Matrix* mat, const char* name) {
    if (!mat) {
        printf("Matrix %s: NULL\n", name ? name : "");
        return;
    }

    printf("Matrix %s [%zu x %zu]:\n", name ? name : "unnamed", mat->rows, mat->cols);
    for (size_t i = 0; i < mat->rows; ++i) {
        printf("  [ ");
        for (size_t j = 0; j < mat->cols; ++j) {
            printf("%8.4f ", mat->data[i * mat->cols + j]);
        }
        printf("]\n");
    }
    printf("\n");
}

Matrix* matrix_transpose(const Matrix* mat) {
    if (!mat) {
        fprintf(stderr, "[Error] matrix_transpose: Input matrix is NULL\n");
        return NULL;
    }

    Matrix* transposed = matrix_create(mat->cols, mat->rows);
    if (!transposed) return NULL;

    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            transposed->data[j * transposed->cols + i] = mat->data[i * mat->cols + j];
        }
    }

    return transposed;
}
