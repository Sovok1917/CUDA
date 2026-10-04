#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <x86intrin.h>
#include <immintrin.h>

class Matrix4D {
public:
    Matrix4D(int outer_rows, int outer_cols)
    : rows_(outer_rows), cols_(outer_cols) {
        int total_floats = rows_ * cols_ * 16;
        data_ = static_cast<float*>(aligned_alloc(64, total_floats * sizeof(float)));
        zero();
    }

    ~Matrix4D() {
        std::free(data_);
    }

    Matrix4D(const Matrix4D&) = delete;
    Matrix4D& operator=(const Matrix4D&) = delete;

    inline float* row_ptr(int i, int j, int x) {
        return &data_[((i * cols_ + j) * 4 + x) * 4];
    }

    inline const float* row_ptr(int i, int j, int x) const {
        return &data_[((i * cols_ + j) * 4 + x) * 4];
    }

    inline float& at(int i, int j, int x, int y) {
        return data_[((i * cols_ + j) * 4 + x) * 4 + y];
    }

    inline float at(int i, int j, int x, int y) const {
        return data_[((i * cols_ + j) * 4 + x) * 4 + y];
    }

    void zero() {
        int total = rows_ * cols_ * 16;
        std::fill(data_, data_ + total, 0.0f);
    }

    void randomize(unsigned int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(1.0f, 2.0f);
        int total = rows_ * cols_ * 16;
        for (int idx = 0; idx < total; ++idx) {
            data_[idx] = dis(gen);
        }
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    int rows_;
    int cols_;
    float* data_;
};

__attribute__((noinline))
void matmul_base(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const int L = A.rows();
    const int M = A.cols();
    const int N = B.cols();

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int x = 0; x < 4; ++x) {
                for (int r = 0; r < M; ++r) {
                    __m128 acc = _mm_load_ps(C.row_ptr(i, j, x));
                    for (int z = 0; z < 4; ++z) {
                        __m128 a_vec = _mm_set1_ps(A.at(i, r, x, z));
                        __m128 b_vec = _mm_load_ps(B.row_ptr(r, j, z));
                        acc = _mm_fmadd_ps(a_vec, b_vec, acc);
                    }
                    _mm_store_ps(C.row_ptr(i, j, x), acc);
                }
            }
        }
    }
}

__attribute__((noinline))
void matmul_optimized(const Matrix4D& A, const Matrix4D& B, Matrix4D& C, int NB) {
    const int L = A.rows();
    const int M = A.cols();
    const int N = B.cols();

    for (int ib = 0; ib < L; ib += NB) {
        const int i_end = std::min(ib + NB, L);
        for (int kb = 0; kb < M; kb += NB) {
            const int k_end = std::min(kb + NB, M);
            for (int jb = 0; jb < N; jb += NB) {
                const int j_end = std::min(jb + NB, N);

                for (int i = ib; i < i_end; ++i) {
                    for (int k = kb; k < k_end; ++k) {
                        const float* a_ptr = A.row_ptr(i, k, 0);
                        __m256 a01[4];
                        __m256 a23[4];
                        for (int p = 0; p < 4; ++p) {
                            a01[p] = _mm256_set_m128(_mm_set1_ps(a_ptr[4 + p]), _mm_set1_ps(a_ptr[p]));
                            a23[p] = _mm256_set_m128(_mm_set1_ps(a_ptr[12 + p]), _mm_set1_ps(a_ptr[8 + p]));
                        }

                        const float* b_ptr = B.row_ptr(k, jb, 0);
                        float* c_ptr = C.row_ptr(i, jb, 0);

                        int j = jb;
                        for (; j <= j_end - 2; j += 2) {
                            __m256 c01_0 = _mm256_load_ps(c_ptr + 0);
                            __m256 c23_0 = _mm256_load_ps(c_ptr + 8);
                            __m256 c01_1 = _mm256_load_ps(c_ptr + 16);
                            __m256 c23_1 = _mm256_load_ps(c_ptr + 24);

                            #pragma GCC unroll 4
                            for (int p = 0; p < 4; ++p) {
                                const __m256 b0 = _mm256_broadcast_ps((const __m128*)(b_ptr + p * 4));
                                const __m256 b1 = _mm256_broadcast_ps((const __m128*)(b_ptr + 16 + p * 4));

                                c01_0 = _mm256_fmadd_ps(a01[p], b0, c01_0);
                                c23_0 = _mm256_fmadd_ps(a23[p], b0, c23_0);
                                c01_1 = _mm256_fmadd_ps(a01[p], b1, c01_1);
                                c23_1 = _mm256_fmadd_ps(a23[p], b1, c23_1);
                            }

                            _mm256_store_ps(c_ptr + 0, c01_0);
                            _mm256_store_ps(c_ptr + 8, c23_0);
                            _mm256_store_ps(c_ptr + 16, c01_1);
                            _mm256_store_ps(c_ptr + 24, c23_1);

                            b_ptr += 32;
                            c_ptr += 32;
                        }

                        for (; j < j_end; ++j) {
                            __m256 c01 = _mm256_load_ps(c_ptr + 0);
                            __m256 c23 = _mm256_load_ps(c_ptr + 8);

                            #pragma GCC unroll 4
                            for (int p = 0; p < 4; ++p) {
                                const __m256 b_row = _mm256_broadcast_ps((const __m128*)(b_ptr + p * 4));
                                c01 = _mm256_fmadd_ps(a01[p], b_row, c01);
                                c23 = _mm256_fmadd_ps(a23[p], b_row, c23);
                            }

                            _mm256_store_ps(c_ptr + 0, c01);
                            _mm256_store_ps(c_ptr + 8, c23);

                            b_ptr += 16;
                            c_ptr += 16;
                        }
                    }
                }

            }
        }
    }
}

bool verify_matrices(const Matrix4D& mat1, const Matrix4D& mat2) {
    if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
        return false;
    }
    for (int i = 0; i < mat1.rows(); ++i) {
        for (int j = 0; j < mat1.cols(); ++j) {
            for (int x = 0; x < 4; ++x) {
                for (int y = 0; y < 4; ++y) {
                    float diff = std::fabs(mat1.at(i, j, x, y) - mat2.at(i, j, x, y));
                    float max_val = std::max(std::fabs(mat1.at(i, j, x, y)), std::fabs(mat2.at(i, j, x, y)));
                    if (diff > 1e-2f && (max_val == 0.0f || diff > 1e-3f * max_val)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static void parse_dimensions(int argc, char* argv[], int& L, int& M, int& N, int& NB) {
    if (argc >= 2) L = std::stoi(argv[1]);
    if (argc >= 3) M = std::stoi(argv[2]);
    if (argc >= 4) N = std::stoi(argv[3]);
    if (argc >= 5) NB = std::stoi(argv[4]);
    if (argc == 2) {
        M = L;
        N = L;
    }
}

int main(int argc, char* argv[]) {
    int L = 1200;
    int M = 1200;
    int N = 1200;
    int NB = 200;

    parse_dimensions(argc, argv, L, M, N, NB);

    std::cout << "Lab 02\n";
    std::cout << "Dimensions: L=" << L << ", M=" << M << ", N=" << N
    << " [" << (L * 4) << "x" << (M * 4) << " * "
    << (M * 4) << "x" << (N * 4) << " -> "
    << (L * 4) << "x" << (N * 4) << "]\n";
    std::cout << "Tile block size: NB=" << NB << " (" << (NB * 4) << "x" << (NB * 4) << " floats, "
    << (NB * 16 * 4) << " bytes wide)\n\n";

    Matrix4D A(L, M);
    Matrix4D B(M, N);
    Matrix4D C1(L, N);
    Matrix4D C2(L, N);

    A.randomize(42);
    B.randomize(84);

    C1.zero();
    uint64_t c_start = __rdtsc();
    auto t_start = std::chrono::high_resolution_clock::now();
    matmul_base(A, B, C1);
    auto t_end = std::chrono::high_resolution_clock::now();
    uint64_t c_end = __rdtsc();
    const double time_base = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_base = c_end - c_start;

    C2.zero();
    c_start = __rdtsc();
    t_start = std::chrono::high_resolution_clock::now();
    matmul_optimized(A, B, C2, NB);
    t_end = std::chrono::high_resolution_clock::now();
    c_end = __rdtsc();
    const double time_opt = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_opt = c_end - c_start;

    const bool matches = verify_matrices(C1, C2);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Result C1 (Base):\n";
    std::cout << "  Time:   " << time_base << " ms (" << (time_base / 1000.0) << " s)\n";

    std::cout << "Result C2 (Optimized):\n";
    std::cout << "  Time:   " << time_opt << " ms (" << (time_opt / 1000.0) << " s)\n";
    std::cout << "  Speedup vs Base: " << (time_base / time_opt) << "x\n";
    std::cout << "  Verification:    " << (matches ? "PASSED" : "FAILED") << "\n\n";

    if (!matches) {
        std::cerr << "Verification error: mismatch in calculated matrices.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
