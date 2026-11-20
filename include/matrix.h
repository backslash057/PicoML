#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} Matrix;

/* Lifecycle & Initializers */
Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_zeros(size_t rows, size_t cols);
Matrix* matrix_ones(size_t rows, size_t cols);
Matrix* matrix_randn(size_t rows, size_t cols, float mean, float std);
Matrix* matrix_from_array(const float* arr, size_t rows, size_t cols);
void matrix_free(Matrix* mat);

/* Utilities */
void matrix_print(const Matrix* mat, const char* name);
Matrix* matrix_transpose(const Matrix* mat);

#endif /* MATRIX_H */
