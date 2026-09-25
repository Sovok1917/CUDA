#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
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

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
__attribute__((noinline))
void matmul_scalar(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const int L = A.rows();
    const int M = A.cols();
    const int N = B.cols();

    for (int i = 0; i < L; ++i) {
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int x = 0; x < 4; ++x) {
                    for (int p = 0; p < 4; ++p) {
                        const float a_val = A.at(i, k, x, p);
                        for (int y = 0; y < 4; ++y) {
                            C.at(i, j, x, y) += a_val * B.at(k, j, p, y);
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
__attribute__((noinline))
void matmul_autovec(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const int L = A.rows();
    const int M = A.cols();
    const int N = B.cols();

    for (int i = 0; i < L; ++i) {
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int x = 0; x < 4; ++x) {
                    for (int p = 0; p < 4; ++p) {
                        const float a_val = A.at(i, k, x, p);
                        const float* b_row = B.row_ptr(k, j, p);
                        float* c_row = C.row_ptr(i, j, x);
                        for (int y = 0; y < 4; ++y) {
                            c_row[y] += a_val * b_row[y];
                        }
                    }
                }
            }
        }
    }
}
#pragma GCC pop_options


#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
__attribute__((noinline))
void matmul_manvec(const Matrix4D& A, const Matrix4D& B, Matrix4D& C) {
    const int L = A.rows();
    const int M = A.cols();
    const int N = B.cols();

    __m256 a[2][4];
    __m256 b[4];
    __m256 c[2];

    for (int i = 0; i < L; ++i) {
        for (int k = 0; k < M; ++k) {
            for (int p = 0; p < 4; ++p) {
                a[0][p] = _mm256_set_m128(_mm_set1_ps(A.at(i, k, 1, p)), _mm_set1_ps(A.at(i, k, 0, p)));
                a[1][p] = _mm256_set_m128(_mm_set1_ps(A.at(i, k, 3, p)), _mm_set1_ps(A.at(i, k, 2, p)));
            }

            for (int j = 0; j < N; ++j) {
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
#pragma GCC pop_options

bool verify_matrices(const Matrix4D& mat1, const Matrix4D& mat2) {
    if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
        return false;
    }
    for (int i = 0; i < mat1.rows(); ++i) {
        for (int j = 0; j < mat1.cols(); ++j) {
            for (int x = 0; x < 4; ++x) {
                for (int y = 0; y < 4; ++y) {
                    if (mat1.at(i, j, x, y) != mat2.at(i, j, x, y)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static void parse_dimensions(int argc, char* argv[], int& L, int& M, int& N) {
    if (argc >= 2) L = std::stoi(argv[1]);
    if (argc >= 3) M = std::stoi(argv[2]);
    if (argc >= 4) N = std::stoi(argv[3]);
    if (argc == 2) {
        M = L;
        N = L;
    }
}

int main(int argc, char* argv[]) {
    int L = 600;
    int M = 400;
    int N = 500;

    parse_dimensions(argc, argv, L, M, N);

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

    A.randomize(42);
    B.randomize(84);

    C1_off.zero();
    uint64_t c_start = __rdtsc();
    auto t_start = std::chrono::high_resolution_clock::now();
    matmul_scalar(A, B, C1_off);
    auto t_end = std::chrono::high_resolution_clock::now();
    uint64_t c_end = __rdtsc();
    const double time_scalar = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_scalar = c_end - c_start;

    C1_on.zero();
    c_start = __rdtsc();
    t_start = std::chrono::high_resolution_clock::now();
    matmul_autovec(A, B, C1_on);
    t_end = std::chrono::high_resolution_clock::now();
    c_end = __rdtsc();
    const double time_autovec = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_autovec = c_end - c_start;

    C2.zero();
    c_start = __rdtsc();
    t_start = std::chrono::high_resolution_clock::now();
    matmul_manvec(A, B, C2);
    t_end = std::chrono::high_resolution_clock::now();
    c_end = __rdtsc();
    const double time_manvec = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const uint64_t cycles_manvec = c_end - c_start;

    const bool autovec_matches = verify_matrices(C1_off, C1_on);
    const bool manvec_matches = verify_matrices(C1_off, C2);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Result C1 (Vectorization OFF):\n";
    std::cout << "  Time:   " << time_scalar << " ms (" << (time_scalar / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_scalar << "\n\n";

    std::cout << "Result C1 (Compiler vectorization):\n";
    std::cout << "  Time:   " << time_autovec << " ms (" << (time_autovec / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_autovec << "\n";
    std::cout << "  Speedup vs OFF: " << (time_scalar / time_autovec) << "x\n";
    std::cout << "  Verification:   " << (autovec_matches ? "PASSED" : "FAILED") << "\n\n";

    std::cout << "Result C2 (Manual vectorization):\n";
    std::cout << "  Time:   " << time_manvec << " ms (" << (time_manvec / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_manvec << "\n";
    std::cout << "  Speedup vs OFF:      " << (time_scalar / time_manvec) << "x\n";
    std::cout << "  Speedup vs Compiler: " << (time_autovec / time_manvec) << "x\n";
    std::cout << "  Verification:        " << (manvec_matches ? "PASSED" : "FAILED") << "\n\n";

    if (!autovec_matches || !manvec_matches) {
        std::cerr << "Verification error: mismatch in calculated matrices.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
