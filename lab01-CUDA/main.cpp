#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <x86intrin.h>
#include <immintrin.h>

/**
 * Purpose: Matrix container representing a 4D block matrix [L x M x 4 x 4].
 *          Uses a single contiguous 64-byte aligned buffer for optimal cache access.
 */
class Matrix4D {
public:
    /**
     * Purpose: Allocates contiguous aligned memory for an [outer_rows x outer_cols x 4 x 4] matrix.
     * Input:   outer_rows - number of outer block rows.
     *          outer_cols - number of outer block cols.
     */
    Matrix4D(size_t outer_rows, size_t outer_cols)
    : rows_(outer_rows), cols_(outer_cols) {
        const size_t total_floats = rows_ * cols_ * 16;
        data_ = static_cast<float*>(aligned_alloc(64, total_floats * sizeof(float)));
        zero();
    }

    ~Matrix4D() {
        std::free(data_);
    }

    Matrix4D(const Matrix4D&) = delete;
    Matrix4D& operator=(const Matrix4D&) = delete;

    /**
     * Purpose: Returns pointer to row x of block (i, j).
     * Input:   i - outer row index.
     *          j - outer col index.
     *          x - inner row index (0..3).
     * Return:  float* - pointer to the first element of that 4-float row.
     */
    inline float* row_ptr(size_t i, size_t j, size_t x) {
        return &data_[((i * cols_ + j) * 4 + x) * 4];
    }

    inline const float* row_ptr(size_t i, size_t j, size_t x) const {
        return &data_[((i * cols_ + j) * 4 + x) * 4];
    }

    /**
     * Purpose: Direct element access: matrix[i][j][x][y].
     * Input:   i - outer row, j - outer col, x - inner row, y - inner col.
     * Return:  float& - reference to element.
     */
    inline float& at(size_t i, size_t j, size_t x, size_t y) {
        return data_[((i * cols_ + j) * 4 + x) * 4 + y];
    }

    inline float at(size_t i, size_t j, size_t x, size_t y) const {
        return data_[((i * cols_ + j) * 4 + x) * 4 + y];
    }

    /**
     * Purpose: Clears all elements to zero.
     * Input:   None.
     * Return:  void.
     */
    void zero() {
        const size_t total = rows_ * cols_ * 16;
        std::fill(data_, data_ + total, 0.0f);
    }

    /**
     * Purpose: Fills matrix with deterministic pseudo-random float values.
     * Input:   seed - random generator seed.
     * Return:  void.
     */
    void randomize(unsigned int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(1.0f, 2.0f);
        const size_t total = rows_ * cols_ * 16;
        for (size_t idx = 0; idx < total; ++idx) {
            data_[idx] = dis(gen);
        }
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    size_t rows_;
    size_t cols_;
    float* data_;
};

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
/**
 * Purpose: Multiplies block matrices A and B with compiler auto-vectorization explicitly disabled.
 * Input:   A - input matrix [L x M x 4 x 4].
 *          B - input matrix [M x N x 4 x 4].
 *          C - pre-zeroed output matrix [L x N x 4 x 4].
 * Return:  void.
 */

void matmul_scalar(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const size_t L = A.rows();
    const size_t M = A.cols();
    const size_t N = B.cols();

    for (size_t i = 0; i < L; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t r = 0; r < M; ++r) {
                for (size_t x = 0; x < 4; ++x) {
                    for (size_t y = 0; y < 4; ++y) {
                        float sum = 0.0f;
                        for (size_t z = 0; z < 4; ++z) {
                            sum += A.at(i, r, x, z) * B.at(r, j, z, y);
                        }
                        C.at(i, j, x, y) += sum;
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
 * Input:   A - input matrix [L x M x 4 x 4].
 *          B - input matrix [M x N x 4 x 4].
 *          C - pre-zeroed output matrix [L x N x 4 x 4].
 * Return:  void.
 */
void matmul_autovec(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const size_t L = A.rows();
    const size_t M = A.cols();
    const size_t N = B.cols();

    for (size_t i = 0; i < L; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t r = 0; r < M; ++r) {
                for (size_t x = 0; x < 4; ++x) {
                    for (size_t z = 0; z < 4; ++z) {
                        const float a_val = A.at(i, r, x, z);
                        const float* b_row = B.row_ptr(r, j, z);
                        float* c_row = C.row_ptr(i, j, x);
                        for (size_t y = 0; y < 4; ++y) {
                            c_row[y] += a_val * b_row[y];
                        }
                    }
                }
            }
        }
    }
}
#pragma GCC pop_options



/**
 * Purpose: Multiplies block matrices using 256-bit AVX2 registers and hardware FMA.
 *          Processes two block rows simultaneously (Row 0+1 and Row 2+3) without transposition.
 * Input:   A - input block matrix [L x M x 4 x 4].
 *          B - input block matrix [M x N x 4 x 4].
 *          C - pre-zeroed output block matrix [L x N x 4 x 4].
 * Return:  void.
 */
void matmul_manvec(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const size_t L = A.rows();
    const size_t M = A.cols();
    const size_t N = B.cols();

    __m256 a[2][4];
    __m256 b[4];
    __m256 c[2];

    for (size_t i = 0; i < L; ++i) {
        for (size_t k = 0; k < M; ++k) {
            for (int p = 0; p < 4; ++p) {
                a[0][p] = _mm256_set_m128(_mm_set1_ps(A.at(i, k, 1, p)), _mm_set1_ps(A.at(i, k, 0, p)));
                a[1][p] = _mm256_set_m128(_mm_set1_ps(A.at(i, k, 3, p)), _mm_set1_ps(A.at(i, k, 2, p)));
            }

            for (size_t j = 0; j < N; ++j) {
                c[0] = _mm256_load_ps(C.row_ptr(i, j, 0));
                c[1] = _mm256_load_ps(C.row_ptr(i, j, 2));

                for (int p = 0; p < 4; ++p) {
                    b[p] = _mm256_broadcast_ps((const __m128*)B.row_ptr(k, j, p));
                    c[0] = _mm256_fmadd_ps(a[0][p], b[p], c[0]);
                    c[1] = _mm256_fmadd_ps(a[1][p], b[p], c[1]);
                }

                _mm256_store_ps(C.row_ptr(i, j, 0), c[0]);
                _mm256_store_ps(C.row_ptr(i, j, 2), c[1]);
            }
        }
    }
}

/**
 * Purpose: Verifies numerical match between two matrices within an epsilon threshold.
 * Input:   mat1    - first matrix.
 *          mat2    - second matrix.
 *          epsilon - relative tolerance threshold.
 * Return:  bool - true if matrices match, false otherwise.
 */
bool verify_matrices(const Matrix4D& mat1, const Matrix4D& mat2, float epsilon = 1e-3f) {
    if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
        return false;
    }
    for (size_t i = 0; i < mat1.rows(); ++i) {
        for (size_t j = 0; j < mat1.cols(); ++j) {
            for (size_t x = 0; x < 4; ++x) {
                for (size_t y = 0; y < 4; ++y) {
                    float v1 = mat1.at(i, j, x, y);
                    float v2 = mat2.at(i, j, x, y);
                    float diff = std::fabs(v1 - v2);
                    float max_v = std::max(std::fabs(v1), std::fabs(v2));
                    if (diff > epsilon && (max_v == 0.0f || diff > epsilon * max_v)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

//.argv - optional matrix dimensions [L [M N]].
int main(int argc, char* argv[]) {
    size_t L = 600;
    size_t M = 400;
    size_t N = 500;

    if (argc >= 2) L = std::stoul(argv[1]);
    if (argc >= 3) M = std::stoul(argv[2]);
    if (argc >= 4) N = std::stoul(argv[3]);
    if (argc == 2) {
        M = L;
        N = L;
    }

    std::cout << "Lab 01 (float, 4x4 blocks)\n";
    std::cout << "Dimensions: L=" << L << ", M=" << M << ", N=" << N
    << " [" << (L * 4) << "x" << (M * 4) << " * "
    << (M * 4) << "x" << (N * 4) << " -> "
    << (L * 4) << "x" << (N * 4) << "]\n\n";

    Matrix4D A(L, M);
    Matrix4D B(M, N);
    Matrix4D C1_off(L, N);
    Matrix4D C1_on(L, N);
    Matrix4D C2(L, N);

    A.randomize(1);
    B.randomize(2);

    C1_off.zero();  //No Vectorization
    uint64_t c_start = __rdtsc();
    auto t_start = std::chrono::high_resolution_clock::now();
    matmul_scalar(A, B, C1_off);
    auto t_end = std::chrono::high_resolution_clock::now();
    uint64_t c_end = __rdtsc();
    const double time_scalar = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_scalar = c_end - c_start;

    C1_on.zero();   //Compiler Vectorization
    c_start = __rdtsc();
    t_start = std::chrono::high_resolution_clock::now();
    matmul_autovec(A, B, C1_on);
    t_end = std::chrono::high_resolution_clock::now();
    c_end = __rdtsc();
    const double time_autovec = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_autovec = c_end - c_start;

    C2.zero();  //Manual Vectorization
    c_start = __rdtsc();
    t_start = std::chrono::high_resolution_clock::now();
    matmul_manvec(A, B, C2);
    t_end = std::chrono::high_resolution_clock::now();
    c_end = __rdtsc();
    const double time_manvec = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_sse = c_end - c_start;

    const bool autovec_matches = verify_matrices(C1_off, C1_on);
    const bool sse_matches = verify_matrices(C1_off, C2);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Result C1 (No Vectorization):\n";
    std::cout << "  Time:   " << time_scalar << " ms (" << (time_scalar / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_scalar << "\n\n";

    std::cout << "Result C1 (Compiler Vectorization):\n";
    std::cout << "  Time:   " << time_autovec << " ms (" << (time_autovec / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_autovec << "\n";
    std::cout << "  Speedup vs OFF: " << (time_scalar / time_autovec) << "x\n";
    std::cout << "  Verification:   " << (autovec_matches ? "PASSED" : "FAILED") << "\n\n";

    std::cout << "Result C2 (Manual Vectorization):\n";
    std::cout << "  Time:   " << time_manvec << " ms (" << (time_manvec / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_sse << "\n";
    std::cout << "  Speedup vs OFF:      " << (time_scalar / time_manvec) << "x\n";
    std::cout << "  Speedup vs Compiler: " << (time_autovec / time_manvec) << "x\n";
    std::cout << "  Verification:        " << (sse_matches ? "PASSED" : "FAILED") << "\n\n";

    if (!autovec_matches || !sse_matches) {
        std::cerr << "Verification error: mismatch in calculated matrices.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
