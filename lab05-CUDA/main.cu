#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>

constexpr int DIM_MIN = 10;
constexpr int DIM_MAX = 8192;
constexpr int BLOCK = 16;
constexpr int GAUSS_RADIUS = 2;
constexpr int EMBOSS_RADIUS = 1;
constexpr int MAX_KERNEL_SIZE = 25;
constexpr float GAUSS_SIGMA = 1.0f;
constexpr float EMBOSS_BIAS = 128.0f;
constexpr int MAX_ALLOWED_DIFF = 1;

__constant__ float c_weights[MAX_KERNEL_SIZE];

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

__host__ __device__ inline uint8_t to_u8(float v) {
    int i = static_cast<int>(floorf(v + 0.5f));
    return static_cast<uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

template <int R>
__global__ void convolve_gray(const uint8_t* src, uint8_t* dst, int W, int H, float bias) {
    constexpr int K = 2 * R + 1;
    constexpr int TILE = BLOCK + 2 * R;
    __shared__ uint8_t tile[TILE][TILE];

    for (int ty = threadIdx.y; ty < TILE; ty += BLOCK) {
        for (int tx = threadIdx.x; tx < TILE; tx += BLOCK) {
            int gx = min(max(blockIdx.x * BLOCK + tx - R, 0), W - 1);
            int gy = min(max(blockIdx.y * BLOCK + ty - R, 0), H - 1);
            tile[ty][tx] = src[gy * W + gx];
        }
    }
    __syncthreads();

    int x = blockIdx.x * BLOCK + threadIdx.x;
    int y = blockIdx.y * BLOCK + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    float sum = bias;
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            sum += c_weights[dy * K + dx] * tile[threadIdx.y + dy][threadIdx.x + dx];
        }
    }
    dst[y * W + x] = to_u8(sum);
}

template <int R>
__global__ void convolve_rgb(const uchar3* src, uchar3* dst, int W, int H, float bias) {
    constexpr int K = 2 * R + 1;
    constexpr int TILE = BLOCK + 2 * R;
    __shared__ uchar3 tile[TILE][TILE];

    for (int ty = threadIdx.y; ty < TILE; ty += BLOCK) {
        for (int tx = threadIdx.x; tx < TILE; tx += BLOCK) {
            int gx = min(max(blockIdx.x * BLOCK + tx - R, 0), W - 1);
            int gy = min(max(blockIdx.y * BLOCK + ty - R, 0), H - 1);
            tile[ty][tx] = src[gy * W + gx];
        }
    }
    __syncthreads();

    int x = blockIdx.x * BLOCK + threadIdx.x;
    int y = blockIdx.y * BLOCK + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    float r = bias;
    float g = bias;
    float b = bias;
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            float w = c_weights[dy * K + dx];
            uchar3 p = tile[threadIdx.y + dy][threadIdx.x + dx];
            r += w * p.x;
            g += w * p.y;
            b += w * p.z;
        }
    }
    dst[y * W + x] = make_uchar3(to_u8(r), to_u8(g), to_u8(b));
}

template <int R, typename T>
void convolve_gpu(const T* src, T* dst, int W, int H, const std::vector<float>& weights, float bias) {
    check(cudaMemcpyToSymbol(c_weights, weights.data(), weights.size() * sizeof(float)));
    const dim3 block(BLOCK, BLOCK);
    const dim3 grid((W + BLOCK - 1) / BLOCK, (H + BLOCK - 1) / BLOCK);
    if constexpr (std::is_same_v<T, uchar3>) {
        convolve_rgb<R><<<grid, block>>>(src, dst, W, H, bias);
    } else {
        convolve_gray<R><<<grid, block>>>(src, dst, W, H, bias);
    }
    check(cudaGetLastError());
}

template <typename T>
void gpu_pipeline(const T* in, T* tmp, T* out, int W, int H,
                  const std::vector<float>& gauss, const std::vector<float>& emboss) {
    convolve_gpu<GAUSS_RADIUS>(in, tmp, W, H, gauss, 0.0f);
    convolve_gpu<EMBOSS_RADIUS>(tmp, out, W, H, emboss, EMBOSS_BIAS);
}

std::vector<float> make_gauss(float sigma) {
    constexpr int K = 2 * GAUSS_RADIUS + 1;
    std::vector<float> w(K * K);
    float sum = 0.0f;
    for (int dy = -GAUSS_RADIUS; dy <= GAUSS_RADIUS; ++dy) {
        for (int dx = -GAUSS_RADIUS; dx <= GAUSS_RADIUS; ++dx) {
            float v = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            w[(dy + GAUSS_RADIUS) * K + (dx + GAUSS_RADIUS)] = v;
            sum += v;
        }
    }
    for (float& v : w) {
        v /= sum;
    }
    return w;
}

std::vector<uint8_t> make_image(int W, int H, int C) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> noise(0, 63);
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * C);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int checker = (((x >> 4) + (y >> 4)) & 1) * 128;
                int ramp = (x + y + c * 21) & 63;
                img[(y * W + x) * C + c] = static_cast<uint8_t>(checker + ramp + noise(rng));
            }
        }
    }
    return img;
}

std::vector<uint8_t> cpu_convolve(const std::vector<uint8_t>& src, int W, int H, int C,
                                int R, const std::vector<float>& weights, float bias) {
    const int K = 2 * R + 1;
    std::vector<uint8_t> dst(src.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = bias;
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        int sx = std::clamp(x + dx, 0, W - 1);
                        int sy = std::clamp(y + dy, 0, H - 1);
                        sum += weights[(dy + R) * K + (dx + R)] * src[(sy * W + sx) * C + c];
                    }
                }
                dst[(y * W + x) * C + c] = to_u8(sum);
            }
        }
    }
    return dst;
}

template <typename T>
bool run(int W, int H, int C, const char* name,
            const std::vector<float>& gauss, const std::vector<float>& emboss) {
    const std::vector<uint8_t> host_in = make_image(W, H, C);

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> blurred = cpu_convolve(host_in, W, H, C, GAUSS_RADIUS, gauss, 0.0f);
    const std::vector<uint8_t> reference = cpu_convolve(blurred, W, H, C, EMBOSS_RADIUS, emboss, EMBOSS_BIAS);
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    const size_t count = static_cast<size_t>(W) * H;
    const size_t bytes = count * sizeof(T);
    T* d_in;
    T* d_tmp;
    T* d_out;
    check(cudaMalloc(&d_in, bytes));
    check(cudaMalloc(&d_tmp, bytes));
    check(cudaMalloc(&d_out, bytes));
    check(cudaMemcpy(d_in, host_in.data(), bytes, cudaMemcpyHostToDevice));

    gpu_pipeline(d_in, d_tmp, d_out, W, H, gauss, emboss);
    check(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    gpu_pipeline(d_in, d_tmp, d_out, W, H, gauss, emboss);
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float gpu_ms = 0.0f;
    check(cudaEventElapsedTime(&gpu_ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    std::vector<uint8_t> host_out(bytes);
    check(cudaMemcpy(host_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(d_in));
    check(cudaFree(d_tmp));
    check(cudaFree(d_out));

    bool match = true;
    for (size_t i = 0; i < host_out.size(); ++i) {
        if (std::abs(static_cast<int>(host_out[i]) - static_cast<int>(reference[i])) > MAX_ALLOWED_DIFF) {
            match = false;
            break;
        }
    }

    const double speedup = gpu_ms > 0.0f ? cpu_ms / gpu_ms : 0.0;
    std::cout << name << ": CPU " << cpu_ms << " ms, GPU " << gpu_ms << " ms, speedup "
    << speedup << "x, " << (match ? "MATCH" : "MISMATCH") << "\n";
    return match;
}

int main(int argc, char* argv[]) {
    int W = 1920;
    int H = 1080;
    if (argc >= 2) {
        W = std::stoi(argv[1]);
        H = W;
    }
    if (argc >= 3) {
        H = std::stoi(argv[2]);
    }
    if (W < DIM_MIN || W > DIM_MAX || H < DIM_MIN || H > DIM_MAX) {
        std::cerr << "W and H must be in [" << DIM_MIN << ", " << DIM_MAX << "]\n";
        return 1;
    }

    const std::vector<float> gauss = make_gauss(GAUSS_SIGMA);
    const std::vector<float> emboss = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

    std::cout << "Lab 05, task 9: Gaussian 5x5 (sigma=1.0) -> Emboss 3x3\n";
    std::cout << "Image: " << W << "x" << H << "\n\n";
    std::cout << std::fixed << std::setprecision(2);

    const bool gray_ok = run<uint8_t>(W, H, 1, "Gray ", gauss, emboss);
    const bool color_ok = run<uchar3>(W, H, 3, "Color", gauss, emboss);
    return (gray_ok && color_ok) ? 0 : 1;
}
