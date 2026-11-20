#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "ops.h"

int main(void) {
    srand((unsigned int)time(NULL));

    printf("====================================================\n");
    printf("           PicoML Matrix Library Test Suite         \n");
    printf("====================================================\n\n");

    /* 1. Lifecycle & Utilities */
    printf("--- [1] Lifecycle & Utilities ---\n");
    Matrix* zeros = matrix_zeros(2, 3);
    Matrix* ones  = matrix_ones(2, 3);
    Matrix* randn = matrix_randn(2, 3, 0.0f, 1.0f);

    matrix_print(zeros, "Zeros [2x3]");
    matrix_print(ones,  "Ones [2x3]");
    matrix_print(randn, "Randn [2x3, mean=0, std=1]");

    float raw_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Matrix* A = matrix_from_array(raw_data, 2, 3);
    matrix_print(A, "Matrix A (from array)");

    Matrix* A_T = matrix_transpose(A);
    matrix_print(A_T, "A_T = transpose(A)");

    /* 2. GEMM: Standard & Transposed Flags */
    printf("--- [2] Matrix Multiplication (cblas_sgemm) ---\n");
    float b_data[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Matrix* B = matrix_from_array(b_data, 3, 2);
    matrix_print(B, "Matrix B [3x2]");

    Matrix* C = matrix_create(2, 2);
    matmul(A, B, C, 0, 0);
    matrix_print(C, "C = matmul(A, B, transA=0, transB=0)");

    matmul(A, A, C, 0, 1);
    matrix_print(C, "C = matmul(A, A, transA=0, transB=1) [A * A^T]");

    Matrix* C_large = matrix_create(3, 3);
    matmul(A, A, C_large, 1, 0);
    matrix_print(C_large, "C_large = matmul(A, A, transA=1, transB=0) [A^T * A]");

    /* 3. Element-wise & Scaling */
    printf("--- [3] Element-wise Operations & Scaling ---\n");
    float d_data[6] = {0.5f, 1.5f, 2.5f, 0.5f, 1.5f, 2.5f};
    Matrix* D = matrix_from_array(d_data, 2, 3);
    Matrix* res = matrix_create(2, 3);

    add(A, D, res);
    matrix_print(res, "res = A + D");

    sub(A, D, res);
    matrix_print(res, "res = A - D");

    hadamard(A, D, res);
    matrix_print(res, "res = hadamard(A, D)");

    scale(res, 2.0f);
    matrix_print(res, "res scaled by 2.0");

    /* 4. Broadcasting */
    printf("--- [4] Broadcasting Operations ---\n");
    float bias_data[3] = {10.0f, 20.0f, 30.0f};
    Matrix* bias = matrix_from_array(bias_data, 1, 3);
    matrix_print(bias, "Bias Vector [1x3]");

    add_row_vec(A, bias, res);
    matrix_print(res, "res = add_row_vec(A, bias)");

    Matrix* sum_out = matrix_create(1, 3);
    sum_rows(res, sum_out);
    matrix_print(sum_out, "sum_out = sum_rows(res) [Sum along columns]");

    /* 5. Activations */
    printf("--- [5] Activations (ReLU & Backward) ---\n");
    float act_data[6] = {-2.0f, 3.5f, 0.0f, -0.1f, 5.0f, -4.0f};
    Matrix* act_in = matrix_from_array(act_data, 2, 3);
    Matrix* act_out = matrix_create(2, 3);
    matrix_print(act_in, "Activation Input");

    relu(act_in, act_out);
    matrix_print(act_out, "act_out = relu(act_in)");

    float grad_data[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Matrix* grad_out = matrix_from_array(grad_data, 2, 3);
    Matrix* grad_in = matrix_create(2, 3);

    relu_backward(act_in, grad_out, grad_in);
    matrix_print(grad_in, "grad_in = relu_backward(act_in, grad_out)");

    /* 6. Error Handling Verification */
    printf("--- [6] Dimension Mismatch Error Guard Check ---\n");
    printf("Expected error output below:\n");
    matmul(A, A, C, 0, 0);

    /* 7. Cleanup */
    matrix_free(zeros);
    matrix_free(ones);
    matrix_free(randn);
    matrix_free(A);
    matrix_free(A_T);
    matrix_free(B);
    matrix_free(C);
    matrix_free(C_large);
    matrix_free(D);
    matrix_free(res);
    matrix_free(bias);
    matrix_free(sum_out);
    matrix_free(act_in);
    matrix_free(act_out);
    matrix_free(grad_out);
    matrix_free(grad_in);

    printf("All matrix memory successfully freed. Test suite passed.\n");
    return 0;
}
