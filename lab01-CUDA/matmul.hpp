#pragma once

#include "matrix.hpp"

/**
 * Purpose: Multiplies block matrices A and B, storing result in C.
 *          Compiler vectorization is explicitly disabled.
 * Input:   a - reference to input block matrix A [L x M].
 *          b - reference to input block matrix B [M x N].
 *          c - reference to output block matrix C [L x N].
 * Return:  void.
 */
void matmul_scalar(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);

/**
 * Purpose: Multiplies block matrices A and B, storing result in C.
 *          Compiled with tree-vectorization enabled for compiler auto-vectorization.
 * Input:   a - reference to input block matrix A [L x M].
 *          b - reference to input block matrix B [M x N].
 *          c - reference to output block matrix C [L x N].
 * Return:  void.
 */
void matmul_autovec(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);

/**
 * Purpose: Multiplies block matrices A and B, storing result in C.
 *          Manually vectorized using SSE2 intrinsics without transposition.
 * Input:   a - reference to input block matrix A [L x M].
 *          b - reference to input block matrix B [M x N].
 *          c - reference to output block matrix C [L x N].
 * Return:  void.
 */
void matmul_sse(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c);
