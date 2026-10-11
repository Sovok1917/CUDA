#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <x86intrin.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

struct ComplexBounds {
    float xmin;
    float xmax;
    float ymin;
    float ymax;
};

__host__ __device__ inline cuComplex map_grid_to_complex(int col, int row, int W, int H, const ComplexBounds& b) {
    float dx = (b.xmax - b.xmin) / static_cast<float>(W - 1);
    float dy = (b.ymax - b.ymin) / static_cast<float>(H - 1);
    float cx = b.xmin + static_cast<float>(col) * dx;
    float cy = b.ymin + static_cast<float>(row) * dy;
    return make_cuComplex(cx, cy);
}

__host__ __device__ inline int compute_mandelbrot_pixel(cuComplex c, int max_iter) {
    cuComplex z = make_cuComplex(0.0f, 0.0f);
    int iter = 0;
    while (iter < max_iter) {
        float r = cuCrealf(z);
        float i = cuCimagf(z);
        if (r * r + i * i > 4.0f) {
            break;
        }
        z = cuCaddf(cuCmulf(z, z), c);
        iter++;
    }
    return iter;
}

void mandelbrot_cpu(int* output, int W, int H, int max_iter, ComplexBounds bounds) {
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            cuComplex pt = map_grid_to_complex(c, r, W, H, bounds);
            output[r * W + c] = compute_mandelbrot_pixel(pt, max_iter);
        }
    }
}

__global__ void mandelbrot_gpu(int* output, int W, int H, int max_iter, ComplexBounds bounds) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < W && r < H) {
        cuComplex pt = map_grid_to_complex(c, r, W, H, bounds);
        output[r * W + c] = compute_mandelbrot_pixel(pt, max_iter);
    }
}

bool verify_results(const int* cpu_res, const int* gpu_res, int total) {
    for (int i = 0; i < total; ++i) {
        if (cpu_res[i] != gpu_res[i]) {
            return false;
        }
    }
    return true;
}

static void parse_args(int argc, char* argv[], int& W, int& H, int& max_iter) {
    if (argc >= 2) W = std::stoi(argv[1]);
    if (argc >= 3) H = std::stoi(argv[2]);
    if (argc >= 4) max_iter = std::stoi(argv[3]);
    if (argc == 2) {
        H = W;
    }
}

int main(int argc, char* argv[]) {
    int W = 2048;
    int H = 2048;
    int max_iter = 500;
    ComplexBounds bounds = {-2.0f, 2.0f, -2.0f, 2.0f};

    parse_args(argc, argv, W, H, max_iter);

    int total_elements = W * H;
    int total_bytes = total_elements * static_cast<int>(sizeof(int));

    std::cout << "Lab 03\n";
    std::cout << "Grid: " << W << "x" << H << ", Max Iterations: " << max_iter << "\n";
    std::cout << "Bounds: [" << bounds.xmin << ", " << bounds.xmax << "] x ["
    << bounds.ymin << ", " << bounds.ymax << "]\n\n";

    int* h_cpu_output = static_cast<int*>(std::malloc(total_bytes));
    int* h_gpu_output = static_cast<int*>(std::malloc(total_bytes));

    unsigned long long tsc_start = __rdtsc();
    auto t_start = std::chrono::high_resolution_clock::now();
    mandelbrot_cpu(h_cpu_output, W, H, max_iter, bounds);
    auto t_end = std::chrono::high_resolution_clock::now();
    unsigned long long tsc_end = __rdtsc();

    const double time_cpu_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    const unsigned long long cycles_cpu = tsc_end - tsc_start;

    int* d_output = nullptr;
    cudaMalloc(&d_output, total_bytes);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    mandelbrot_gpu<<<grid, block>>>(d_output, W, H, max_iter, bounds);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float time_gpu_raw_ms = 0.0f;
    cudaEventElapsedTime(&time_gpu_raw_ms, start_event, stop_event);
    const double time_gpu_ms = static_cast<double>(time_gpu_raw_ms);

    cudaMemcpy(h_gpu_output, d_output, total_bytes, cudaMemcpyDeviceToHost);

    const bool matches = verify_results(h_cpu_output, h_gpu_output, total_elements);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Result C1 (CPU):\n";
    std::cout << "  Time:   " << time_cpu_ms << " ms (" << (time_cpu_ms / 1000.0) << " s)\n";
    std::cout << "  Cycles: " << cycles_cpu << "\n\n";

    std::cout << "Result C2 (GPU):\n";
    std::cout << "  Time:   " << time_gpu_ms << " ms (" << (time_gpu_ms / 1000.0) << " s)\n";
    std::cout << "  Speedup vs CPU: " << (time_cpu_ms / time_gpu_ms) << "x\n";
    std::cout << "  Verification:   " << (matches ? "PASSED" : "FAILED") << "\n\n";

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_output);
    std::free(h_cpu_output);
    std::free(h_gpu_output);

    if (!matches) {
        std::cerr << "Verification error: mismatch in calculated matrices.\n";
        return 1;
    }

    std::cout << "All results match successfully.\n";
    return 0;
}
