#include "matmul.hpp"
#include "timer.hpp"
#include <iostream>
#include <iomanip>
#include <string>

/**
 * Purpose: Parses CLI arguments for dynamic matrix dimensions L, M, and N.
 * Input:   argc - argument count.
 *          argv - argument vector.
 *          L    - reference to outer row count of matrix A.
 *          M    - reference to outer shared dimension.
 *          N    - reference to outer column count of matrix B.
 * Return:  void.
 */
static void parse_dimensions(int argc, char* argv[], size_t& L, size_t& M, size_t& N) {
    if (argc >= 2) L = std::stoul(argv[1]);
    if (argc >= 3) M = std::stoul(argv[2]);
    if (argc >= 4) N = std::stoul(argv[3]);
    if (argc == 2) {
        M = L;
        N = L;
    }
}

/**
 * Purpose: Entry point. Executes and times multiplication algorithms.
 * Input:   argc - argument count.
 *          argv - argument values [optional: L [M N]].
 * Return:  int - exit status code.
 */
int main(int argc, char* argv[]) {
    size_t L = 600;
    size_t M = 400;
    size_t N = 500;

    parse_dimensions(argc, argv, L, M, N);

    std::cout << "Lab 01 (float, 4x4 blocks)\n";
    std::cout << "Dimensions: L=" << L << ", M=" << M << ", N=" << N
    << " [" << (L * 4) << "x" << (M * 4) << " * "
    << (M * 4) << "x" << (N * 4) << " -> "
    << (L * 4) << "x" << (N * 4) << "]\n\n";

    BlockMatrix A(L, M);
    BlockMatrix B(M, N);
    BlockMatrix C_scalar(L, N);
    BlockMatrix C_autovec(L, N);
    BlockMatrix C_avx2(L, N);

    A.randomize(-1.0f, 1.0f, 100);
    B.randomize(-1.0f, 1.0f, 200);

    Timer timer;

    // 1. Result C1 (Vectorization OFF)
    C_scalar.zero();
    timer.start();
    matmul_scalar(A, B, C_scalar);
    timer.stop();
    const double time_scalar = timer.elapsed_ms();
    const uint64_t cycles_scalar = timer.elapsed_cycles();

    // 2. Result C1 (Compiler vectorization)
    C_autovec.zero();
    timer.start();
    matmul_autovec(A, B, C_autovec);
    timer.stop();
    const double time_autovec = timer.elapsed_ms();
    const uint64_t cycles_autovec = timer.elapsed_cycles();

    // 3. Result C2 (Manual AVX2/FMA vectorization)
    C_avx2.zero();
    timer.start();
    matmul_avx2(A, B, C_avx2);
    timer.stop();
    const double time_avx2 = timer.elapsed_ms();
    const uint64_t cycles_avx2 = timer.elapsed_cycles();

    const bool autovec_matches = verify_matrices(C_scalar, C_autovec);
    const bool avx2_matches = verify_matrices(C_scalar, C_avx2);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Result C1 (Vectorization OFF):\n";
    std::cout << "  Time:   " << time_scalar << " ms (" << (time_scalar / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_scalar << "\n\n";

    std::cout << "Result C1 (Compiler vectorization):\n";
    std::cout << "  Time:   " << time_autovec << " ms (" << (time_autovec / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_autovec << "\n";
    std::cout << "  Speedup vs OFF: " << (time_scalar / time_autovec) << "x\n";
    std::cout << "  Verification:   " << (autovec_matches ? "PASSED" : "FAILED") << "\n\n";

    std::cout << "Result C2 (Manual AVX2/FMA vectorization):\n";
    std::cout << "  Time:   " << time_avx2 << " ms (" << (time_avx2 / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_avx2 << "\n";
    std::cout << "  Speedup vs OFF:      " << (time_scalar / time_avx2) << "x\n";
    std::cout << "  Speedup vs Compiler: " << (time_autovec / time_avx2) << "x\n";
    std::cout << "  Verification:        " << (avx2_matches ? "PASSED" : "FAILED") << "\n\n";

    if (!autovec_matches || !avx2_matches) {
        std::cerr << "Verification error: mismatch in calculated matrices.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
