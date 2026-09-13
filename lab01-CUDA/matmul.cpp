#include "matmul.hpp"
#include <emmintrin.h>

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
/**
 * Purpose: Multiplies block matrices A and B without vectorization.
 * Input:   a - input block matrix A [L x M].
 *          b - input block matrix B [M x N].
 *          c - pre-zeroed output block matrix C [L x N].
 * Return:  void.
 */
__attribute__((noinline))
void matmul_scalar(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c) {
    const size_t L = a.rows();
    const size_t M = a.cols();
    const size_t N = b.cols();

    for (size_t i = 0; i < L; ++i) {
        for (size_t k = 0; k < M; ++k) {
            const auto& a_block = a.at(i, k);
            for (size_t j = 0; j < N; ++j) {
                const auto& b_block = b.at(k, j);
                auto& c_block = c.at(i, j);

                for (size_t r = 0; r < 4; ++r) {
                    for (size_t p = 0; p < 4; ++p) {
                        const float a_val = a_block.data[r][p];
                        for (size_t col = 0; col < 4; ++col) {
                            c_block.data[r][col] += a_val * b_block.data[p][col];
                        }
                    }
                }
            }
        }
    }
}
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC optimize("tree-vectorize")
/**
 * Purpose: Multiplies block matrices A and B with compiler auto-vectorization enabled.
 * Input:   a - input block matrix A [L x M].
 *          b - input block matrix B [M x N].
 *          c - pre-zeroed output block matrix C [L x N].
 * Return:  void.
 */
__attribute__((noinline))
void matmul_autovec(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c) {
    const size_t L = a.rows();
    const size_t M = a.cols();
    const size_t N = b.cols();

    for (size_t i = 0; i < L; ++i) {
        for (size_t k = 0; k < M; ++k) {
            const auto& a_block = a.at(i, k);
            for (size_t j = 0; j < N; ++j) {
                const auto& b_block = b.at(k, j);
                auto& c_block = c.at(i, j);

                for (size_t r = 0; r < 4; ++r) {
                    for (size_t p = 0; p < 4; ++p) {
                        const float a_val = a_block.data[r][p];
                        for (size_t col = 0; col < 4; ++col) {
                            c_block.data[r][col] += a_val * b_block.data[p][col];
                        }
                    }
                }
            }
        }
    }
}
#pragma GCC pop_options

/**
 * Purpose: Multiplies block matrices A and B using manual SSE2 intrinsics without transposition.
 * Input:   a - input block matrix A [L x M].
 *          b - input block matrix B [M x N].
 *          c - pre-zeroed output block matrix C [L x N].
 * Return:  void.
 */
__attribute__((noinline))
void matmul_sse(const BlockMatrix& a, const BlockMatrix& b, BlockMatrix& c) {
    const size_t L = a.rows();
    const size_t M = a.cols();
    const size_t N = b.cols();

    for (size_t i = 0; i < L; ++i) {
        for (size_t k = 0; k < M; ++k) {
            const auto& a_block = a.at(i, k);
            for (size_t j = 0; j < N; ++j) {
                const auto& b_block = b.at(k, j);
                auto& c_block = c.at(i, j);

                const __m128 b_row0 = _mm_load_ps(&b_block.data[0][0]);
                const __m128 b_row1 = _mm_load_ps(&b_block.data[1][0]);
                const __m128 b_row2 = _mm_load_ps(&b_block.data[2][0]);
                const __m128 b_row3 = _mm_load_ps(&b_block.data[3][0]);

                for (size_t r = 0; r < 4; ++r) {
                    __m128 c_row = _mm_load_ps(&c_block.data[r][0]);

                    const __m128 a0 = _mm_set1_ps(a_block.data[r][0]);
                    const __m128 a1 = _mm_set1_ps(a_block.data[r][1]);
                    const __m128 a2 = _mm_set1_ps(a_block.data[r][2]);
                    const __m128 a3 = _mm_set1_ps(a_block.data[r][3]);

                    c_row = _mm_add_ps(c_row, _mm_mul_ps(a0, b_row0));
                    c_row = _mm_add_ps(c_row, _mm_mul_ps(a1, b_row1));
                    c_row = _mm_add_ps(c_row, _mm_mul_ps(a2, b_row2));
                    c_row = _mm_add_ps(c_row, _mm_mul_ps(a3, b_row3));

                    _mm_store_ps(&c_block.data[r][0], c_row);
                }
            }
        }
    }
}
