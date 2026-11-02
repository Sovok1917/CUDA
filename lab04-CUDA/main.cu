#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <random>
#include <x86intrin.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int N_MIN = 100;
constexpr int N_MAX = 10000;
constexpr unsigned int FULL_MASK = 0xffffffffu;

__host__ __device__ inline int is_inversion(int left, int right) {
    return left > right;
}

__device__ inline int warp_reduce_sum(int v) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__device__ inline int block_reduce_sum(int v, int* warp_sums) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    int val = 0;
    if (threadIdx.x < WARPS_PER_BLOCK) {
        val = warp_sums[threadIdx.x];
    }

    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void inversions_gpu(const int* a, int N, int* result) {
    if (blockIdx.y < blockIdx.x) {
        return;
    }

    __shared__ int tile[BLOCK_SIZE];
    __shared__ int warp_sums[WARPS_PER_BLOCK];

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int j0 = blockIdx.y * BLOCK_SIZE;
    int j = j0 + threadIdx.x;

    if (j < N) {
        tile[threadIdx.x] = a[j];
    } else {
        tile[threadIdx.x] = 0;
    }

    int ai = 0;
    if (i < N) {
        ai = a[i];
    }
    __syncthreads();

    int limit = BLOCK_SIZE;
    if (N - j0 < BLOCK_SIZE) {
        limit = N - j0;
    }

    int start = limit;
    if (i < N) {
        if (blockIdx.x == blockIdx.y) {
            start = threadIdx.x + 1;
        } else {
            start = 0;
        }
    }

    int count = 0;
    for (int k = start; k < limit; ++k) {
        count += is_inversion(ai, tile[k]);
    }

    int block_sum = block_reduce_sum(count, warp_sums);
    if (threadIdx.x == 0 && block_sum != 0) {
        atomicAdd(result, block_sum);
    }
}

int inversions_cpu(const int* a, int N) {
    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            count += is_inversion(a[i], a[j]);
        }
    }
    return count;
}

static dim3 make_grid(int N) {
    int tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return dim3(tiles, tiles);
}

static void generate_permutation(int* a, int N, unsigned int seed) {
    std::mt19937 rng(seed);
    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }
    std::shuffle(a, a + N, rng);
}

static int parse_args(int argc, char* argv[]) {
    int N = N_MAX;
    if (argc >= 2) {
        N = std::stoi(argv[1]);
    }
    if (N < N_MIN || N > N_MAX) {
        std::cerr << "N must be in [" << N_MIN << ", " << N_MAX << "]\n";
        std::exit(1);
    }
    return N;
}

int main(int argc, char* argv[]) {
    int N = parse_args(argc, argv);
    int bytes = N * static_cast<int>(sizeof(int));
    dim3 grid = make_grid(N);

    std::cout << "Lab 04\n";
    std::cout << "Task 9: inversions count, N: " << N << "\n";
    std::cout << "Block: " << BLOCK_SIZE << ", Grid: " << grid.x << "x" << grid.y << "\n\n";

    int* h_a = static_cast<int*>(std::malloc(bytes));
    generate_permutation(h_a, N, 12345u);

    unsigned long long tsc_start = __rdtsc();
    auto t_start = std::chrono::high_resolution_clock::now();
    int result_cpu = inversions_cpu(h_a, N);
    auto t_end = std::chrono::high_resolution_clock::now();
    unsigned long long tsc_end = __rdtsc();

    const double time_cpu_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const unsigned long long cycles_cpu = tsc_end - tsc_start;

    int* d_a = nullptr;
    int* d_result = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    cudaMemset(d_result, 0, sizeof(int));
    inversions_gpu<<<grid, BLOCK_SIZE>>>(d_a, N, d_result);
    cudaDeviceSynchronize();
    cudaMemset(d_result, 0, sizeof(int));

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    inversions_gpu<<<grid, BLOCK_SIZE>>>(d_a, N, d_result);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float time_gpu_raw_ms = 0.0f;
    cudaEventElapsedTime(&time_gpu_raw_ms, start_event, stop_event);
    const double time_gpu_ms = static_cast<double>(time_gpu_raw_ms);

    int result_gpu = 0;
    cudaMemcpy(&result_gpu, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    const bool matches = (result_cpu == result_gpu);

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Result C1 (CPU):\n";
    std::cout << "  Inversions: " << result_cpu << "\n";
    std::cout << "  Time:   " << time_cpu_ms << " ms (" << (time_cpu_ms / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_cpu << "\n\n";

    std::cout << "Result C2 (GPU):\n";
    std::cout << "  Inversions: " << result_gpu << "\n";
    std::cout << "  Time:   " << time_gpu_ms << " ms (" << (time_gpu_ms / 1000.0) << " s)\n";
    std::cout << "  Speedup vs CPU: " << (time_cpu_ms / time_gpu_ms) << "x\n";
    std::cout << "  Verification:   " << (matches ? "PASSED" : "FAILED") << "\n\n";

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_a);
    cudaFree(d_result);
    std::free(h_a);

    if (!matches) {
        std::cerr << "Verification error: results do not match.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
