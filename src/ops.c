#include "ops.h"

#include <stdio.h>
#include <string.h>
#include <cblas.h>

void matmul(const Matrix* A, const Matrix* B, Matrix* C, int transA, int transB) {
    if (!A || !B || !C) {
        fprintf(stderr, "[Error] matmul: Null pointer encountered (A=%p, B=%p, C=%p)\n",
                (void*)A, (void*)B, (void*)C);
        return;
    }

    size_t opA_rows = transA ? A->cols : A->rows;
    size_t opA_cols = transA ? A->rows : A->cols;
    size_t opB_rows = transB ? B->cols : B->rows;
    size_t opB_cols = transB ? B->rows : B->cols;

    if (opA_cols != opB_rows) {
        fprintf(stderr, "[Error] matmul: Inner dimensions do not match: "
                        "op(A) [%zu x %zu] vs op(B) [%zu x %zu]\n",
                opA_rows, opA_cols, opB_rows, opB_cols);
        return;
    }

    if (C->rows != opA_rows || C->cols != opB_cols) {
        fprintf(stderr, "[Error] matmul: Destination C shape [%zu x %zu] does not match expected [%zu x %zu]\n",
                C->rows, C->cols, opA_rows, opB_cols);
        return;
    }

    CBLAS_TRANSPOSE tA = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tB = transB ? CblasTrans : CblasNoTrans;

    int M = (int)opA_rows;
    int N = (int)opB_cols;
    int K = (int)opA_cols;
    int lda = (int)A->cols;
    int ldb = (int)B->cols;
    int ldc = (int)C->cols;

    cblas_sgemm(CblasRowMajor, tA, tB,
                M, N, K,
                1.0f,
                A->data, lda,
                B->data, ldb,
                0.0f,
                C->data, ldc);
}

void add(const Matrix* A, const Matrix* B, Matrix* C) {
    if (!A || !B || !C) {
        fprintf(stderr, "[Error] add: Null pointer encountered\n");
        return;
    }

    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != C->rows || A->cols != C->cols) {
        fprintf(stderr, "[Error] add: Dimension mismatch between A [%zu x %zu], B [%zu x %zu], C [%zu x %zu]\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        return;
    }

    int total = (int)(A->rows * A->cols);

    if (C != A) {
        memcpy(C->data, A->data, total * sizeof(float));
    }
    cblas_saxpy(total, 1.0f, B->data, 1, C->data, 1);
}

void sub(const Matrix* A, const Matrix* B, Matrix* C) {
    if (!A || !B || !C) {
        fprintf(stderr, "[Error] sub: Null pointer encountered\n");
        return;
    }

    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != C->rows || A->cols != C->cols) {
        fprintf(stderr, "[Error] sub: Dimension mismatch between A [%zu x %zu], B [%zu x %zu], C [%zu x %zu]\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        return;
    }

    int total = (int)(A->rows * A->cols);

    if (C == B && C != A) {
        cblas_sscal(total, -1.0f, C->data, 1);
        cblas_saxpy(total, 1.0f, A->data, 1, C->data, 1);
    } else {
        if (C != A) {
            memcpy(C->data, A->data, total * sizeof(float));
        }
        cblas_saxpy(total, -1.0f, B->data, 1, C->data, 1);
    }
}

void hadamard(const Matrix* A, const Matrix* B, Matrix* C) {
    if (!A || !B || !C) {
        fprintf(stderr, "[Error] hadamard: Null pointer encountered\n");
        return;
    }

    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != C->rows || A->cols != C->cols) {
        fprintf(stderr, "[Error] hadamard: Dimension mismatch between A [%zu x %zu], B [%zu x %zu], C [%zu x %zu]\n",
                A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        return;
    }

    size_t total = A->rows * A->cols;
    for (size_t i = 0; i < total; ++i) {
        C->data[i] = A->data[i] * B->data[i];
    }
}

void scale(Matrix* A, float scalar) {
    if (!A) {
        fprintf(stderr, "[Error] scale: Matrix pointer is NULL\n");
        return;
    }

    int total = (int)(A->rows * A->cols);
    cblas_sscal(total, scalar, A->data, 1);
}

void add_row_vec(const Matrix* A, const Matrix* row_vec, Matrix* C) {
    if (!A || !row_vec || !C) {
        fprintf(stderr, "[Error] add_row_vec: Null pointer encountered\n");
        return;
    }

    if (row_vec->rows != 1 || row_vec->cols != A->cols) {
        fprintf(stderr, "[Error] add_row_vec: row_vec shape [%zu x %zu] must be [1 x %zu]\n",
                row_vec->rows, row_vec->cols, A->cols);
        return;
    }

    if (C->rows != A->rows || C->cols != A->cols) {
        fprintf(stderr, "[Error] add_row_vec: C shape [%zu x %zu] does not match A [%zu x %zu]\n",
                C->rows, C->cols, A->rows, A->cols);
        return;
    }

    int cols = (int)A->cols;
    for (size_t i = 0; i < A->rows; ++i) {
        float* c_row = C->data + i * cols;
        const float* a_row = A->data + i * cols;

        if (C != A) {
            memcpy(c_row, a_row, cols * sizeof(float));
        }
        cblas_saxpy(cols, 1.0f, row_vec->data, 1, c_row, 1);
    }
}

void sum_rows(const Matrix* A, Matrix* row_vec) {
    if (!A || !row_vec) {
        fprintf(stderr, "[Error] sum_rows: Null pointer encountered\n");
        return;
    }

    if (row_vec->rows != 1 || row_vec->cols != A->cols) {
        fprintf(stderr, "[Error] sum_rows: Destination shape [%zu x %zu] must be [1 x %zu]\n",
                row_vec->rows, row_vec->cols, A->cols);
        return;
    }

    int cols = (int)A->cols;
    memset(row_vec->data, 0, cols * sizeof(float));

    for (size_t i = 0; i < A->rows; ++i) {
        const float* a_row = A->data + i * cols;
        cblas_saxpy(cols, 1.0f, a_row, 1, row_vec->data, 1);
    }
}

void relu(const Matrix* A, Matrix* C) {
    if (!A || !C) {
        fprintf(stderr, "[Error] relu: Null pointer encountered\n");
        return;
    }

    if (A->rows != C->rows || A->cols != C->cols) {
        fprintf(stderr, "[Error] relu: Shape mismatch between A [%zu x %zu] and C [%zu x %zu]\n",
                A->rows, A->cols, C->rows, C->cols);
        return;
    }

    size_t total = A->rows * A->cols;
    for (size_t i = 0; i < total; ++i) {
        C->data[i] = (A->data[i] > 0.0f) ? A->data[i] : 0.0f;
    }
}

void relu_backward(const Matrix* A, const Matrix* grad_out, Matrix* grad_in) {
    if (!A || !grad_out || !grad_in) {
        fprintf(stderr, "[Error] relu_backward: Null pointer encountered\n");
        return;
    }

    if (A->rows != grad_out->rows || A->cols != grad_out->cols ||
        A->rows != grad_in->rows  || A->cols != grad_in->cols) {
        fprintf(stderr, "[Error] relu_backward: Dimension mismatch across A, grad_out, or grad_in\n");
        return;
    }

    size_t total = A->rows * A->cols;
    for (size_t i = 0; i < total; ++i) {
        grad_in->data[i] = (A->data[i] > 0.0f) ? grad_out->data[i] : 0.0f;
    }
}
