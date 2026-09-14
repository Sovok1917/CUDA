#include <immintrin.h>

/**
 * Purpose: Multiplies block matrices using 256-bit AVX2 registers and hardware FMA.
 * Input:   A - input block matrix [L x M x 4 x 4].
 *          B - input block matrix [M x N x 4 x 4].
 *          C - pre-zeroed output block matrix [L x N x 4 x 4].
 * Return:  void.
 */
__attribute__((noinline))
void matmul_manvec(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const size_t L = A.rows();                                                  // Outer rows of A and C
    const size_t M = A.cols();                                                  // Shared outer dimension between A and B
    const size_t N = B.cols();                                                  // Outer columns of B and C

    for (size_t i = 0; i < L; ++i) {                                            // Loop outer rows of A
        for (size_t k = 0; k < M; ++k) {                                        // Loop shared outer blocks
            __m256 a01_p[4];                                                    // Broadcasted weights of block A for rows 0 and 1
            __m256 a23_p[4];                                                    // Broadcasted weights of block A for rows 2 and 3

            for (int p = 0; p < 4; ++p) {                                       // Unroll block multiplication along inner dimension p
                const __m128 a0 = _mm_set1_ps(A.at(i, k, 0, p));                // Duplicate scalar A[0][p] across 4 floats
                const __m128 a1 = _mm_set1_ps(A.at(i, k, 1, p));                // Duplicate scalar A[1][p] across 4 floats
                a01_p[p] = _mm256_set_m128(a1, a0);                             // Combine row 0 (low 128) and row 1 (high 128)

                const __m128 a2 = _mm_set1_ps(A.at(i, k, 2, p));                // Duplicate scalar A[2][p] across 4 floats
                const __m128 a3 = _mm_set1_ps(A.at(i, k, 3, p));                // Duplicate scalar A[3][p] across 4 floats
                a23_p[p] = _mm256_set_m128(a3, a2);                             // Combine row 2 (low 128) and row 3 (high 128)
            }

            for (size_t j = 0; j < N; ++j) {                                    // Stream sequentially through outer columns of B and C
                const __m256 b0 = _mm256_broadcast_ps((const __m128*)B.row_ptr(k, j, 0)); // Replicate row 0 of B across both 128-bit halves
                const __m256 b1 = _mm256_broadcast_ps((const __m128*)B.row_ptr(k, j, 1)); // Replicate row 1 of B across both 128-bit halves
                const __m256 b2 = _mm256_broadcast_ps((const __m128*)B.row_ptr(k, j, 2)); // Replicate row 2 of B across both 128-bit halves
                const __m256 b3 = _mm256_broadcast_ps((const __m128*)B.row_ptr(k, j, 3)); // Replicate row 3 of B across both 128-bit halves

                __m256 c01 = _mm256_load_ps(C.row_ptr(i, j, 0));               // Load 8 contiguous floats of C (rows 0 and 1)
                c01 = _mm256_fmadd_ps(a01_p[0], b0, c01);                       // Multiply and add: c01 += a01_p[0] * b0
                c01 = _mm256_fmadd_ps(a01_p[1], b1, c01);                       // Multiply and add: c01 += a01_p[1] * b1
                c01 = _mm256_fmadd_ps(a01_p[2], b2, c01);                       // Multiply and add: c01 += a01_p[2] * b2
                c01 = _mm256_fmadd_ps(a01_p[3], b3, c01);                       // Multiply and add: c01 += a01_p[3] * b3
                _mm256_store_ps(C.row_ptr(i, j, 0), c01);                       // Store updated rows 0 and 1 back to memory

                __m256 c23 = _mm256_load_ps(C.row_ptr(i, j, 2));               // Load 8 contiguous floats of C (rows 2 and 3)
                c23 = _mm256_fmadd_ps(a23_p[0], b0, c23);                       // Multiply and add: c23 += a23_p[0] * b0
                c23 = _mm256_fmadd_ps(a23_p[1], b1, c23);                       // Multiply and add: c23 += a23_p[1] * b1
                c23 = _mm256_fmadd_ps(a23_p[2], b2, c23);                       // Multiply and add: c23 += a23_p[2] * b2
                c23 = _mm256_fmadd_ps(a23_p[3], b3, c23);                       // Multiply and add: c23 += a23_p[3] * b3
                _mm256_store_ps(C.row_ptr(i, j, 2), c23);                       // Store updated rows 2 and 3 back to memory
            }
        }
    }
}
