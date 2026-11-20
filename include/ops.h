#ifndef OPS_H
#define OPS_H

#include "matrix.h"

/* Matrix Multiplication */
void matmul(const Matrix* A, const Matrix* B, Matrix* C, int transA, int transB);

/* Element-wise & Vectorized Operations */
void add(const Matrix* A, const Matrix* B, Matrix* C);
void sub(const Matrix* A, const Matrix* B, Matrix* C);
void hadamard(const Matrix* A, const Matrix* B, Matrix* C);
void scale(Matrix* A, float scalar);

/* Broadcasting Operations */
void add_row_vec(const Matrix* A, const Matrix* row_vec, Matrix* C);
void sum_rows(const Matrix* A, Matrix* row_vec);

/* Activation Functions */
void relu(const Matrix* A, Matrix* C);
void relu_backward(const Matrix* A, const Matrix* grad_out, Matrix* grad_in);

#endif /* OPS_H */
