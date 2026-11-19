#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

constexpr int DIM_MIN = 10;
constexpr int DIM_MAX = 8192;

constexpr int GRAY_BLOCK_X = 32;
constexpr int GRAY_BLOCK_Y = 8;
constexpr int RGBA_BLOCK_X = 16;
constexpr int RGBA_BLOCK_Y = 16;

constexpr int GAUSS_RADIUS = 2;
constexpr int GAUSS_SIZE = 2 * GAUSS_RADIUS + 1;
constexpr float GAUSS_SIGMA = 1.0f;

constexpr int EMBOSS_SIZE = 3;
constexpr int EMBOSS_BIAS = 128;
constexpr int MAX_ALLOWED_DIFF = 1;

const std::array<int, EMBOSS_SIZE * EMBOSS_SIZE> EMBOSS_KERNEL = {
    -2, -1, 0,
    -1,  1, 1,
    0,  1, 2
};

__constant__ float c_gauss[GAUSS_SIZE * GAUSS_SIZE];
__constant__ int c_emboss[EMBOSS_SIZE * EMBOSS_SIZE];

using GaussWeights = std::array<float, GAUSS_SIZE * GAUSS_SIZE>;
using Plane = std::vector<uint8_t>;

struct Layout {
    int pitch;
    int pixel_stride;
    int channels;
};

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

static int div_up(int a, int b) {
    return (a + b - 1) / b;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) { check(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    uint32_t* get() const { return ptr_; }

private:
    uint32_t* ptr_ = nullptr;
};

__host__ __device__ inline uint8_t clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

struct Window {
    const uint8_t* center;
    int row_stride;
    int col_stride;

    __device__ int at(int dy, int dx) const {
        return center[dy * row_stride + dx * col_stride];
    }
};

struct GaussOp {
    static constexpr int radius = GAUSS_RADIUS;

    __device__ static uint8_t apply(const Window& w) {
        float sum = 0.0f;
        for (int dy = -GAUSS_RADIUS; dy <= GAUSS_RADIUS; ++dy) {
            for (int dx = -GAUSS_RADIUS; dx <= GAUSS_RADIUS; ++dx) {
                sum += c_gauss[(dy + GAUSS_RADIUS) * GAUSS_SIZE + (dx + GAUSS_RADIUS)] * w.at(dy, dx);
            }
        }
        return clamp_u8(static_cast<int>(sum + 0.5f));
    }
};

struct EmbossOp {
    static constexpr int radius = 1;

    __device__ static uint8_t apply(const Window& w) {
        int sum = EMBOSS_BIAS;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                sum += c_emboss[(dy + 1) * EMBOSS_SIZE + (dx + 1)] * w.at(dy, dx);
            }
        }
        return clamp_u8(sum);
    }
};

__device__ inline uint8_t synthetic_value(int x, int y, int channel) {
    unsigned int h = static_cast<unsigned int>(x) * 73856093u
    ^ static_cast<unsigned int>(y) * 19349663u
    ^ static_cast<unsigned int>(channel) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;

    int checker = (((x >> 4) + (y >> 4)) & 1) * 128;
    int ramp = (x + y + channel * 21) & 63;
    int noise = static_cast<int>(h & 63u);
    return static_cast<uint8_t>(checker + ramp + noise);
}

__global__ void generate_gray(uint32_t* img, int W, int H, int words_per_row) {
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (word_x >= words_per_row || y >= H) {
        return;
    }

    uint32_t word = 0;
    for (int k = 0; k < 4; ++k) {
        int x = word_x * 4 + k;
        if (x < W) {
            word |= static_cast<uint32_t>(synthetic_value(x, y, 0)) << (8 * k);
        }
    }
    img[y * words_per_row + word_x] = word;
}

__global__ void generate_rgba(uint32_t* img, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    uint32_t r = synthetic_value(x, y, 0);
    uint32_t g = synthetic_value(x, y, 1);
    uint32_t b = synthetic_value(x, y, 2);
    img[y * W + x] = r | (g << 8) | (b << 16) | 0xFF000000u;
}

__device__ inline uint32_t load_word_clamped(const uint32_t* img, int words_per_row,
                                             int W, int H, int x, int y) {
    y = min(max(y, 0), H - 1);
    if (x >= 0 && x + 3 < W) {
        return img[y * words_per_row + x / 4];
    }

    const uint8_t* row = reinterpret_cast<const uint8_t*>(img) + y * words_per_row * 4;
    uint32_t word = 0;
    for (int k = 0; k < 4; ++k) {
        int px = min(max(x + k, 0), W - 1);
        word |= static_cast<uint32_t>(row[px]) << (8 * k);
    }
    return word;
}

template <class Op>
__global__ void filter_gray(const uint32_t* in, uint32_t* out, int W, int H, int words_per_row) {
    constexpr int R = Op::radius;
    constexpr int TILE_ROWS = GRAY_BLOCK_Y + 2 * R;
    constexpr int TILE_WORDS = GRAY_BLOCK_X + 2;
    constexpr int TILE_STRIDE = TILE_WORDS * 4;

    __shared__ __align__(4) uint8_t tile[TILE_ROWS][TILE_STRIDE];

    const int origin_x = blockIdx.x * GRAY_BLOCK_X * 4 - 4;
    const int origin_y = blockIdx.y * GRAY_BLOCK_Y - R;
    const int tid = threadIdx.y * GRAY_BLOCK_X + threadIdx.x;

    for (int i = tid; i < TILE_ROWS * TILE_WORDS; i += GRAY_BLOCK_X * GRAY_BLOCK_Y) {
        int row = i / TILE_WORDS;
        int col = i % TILE_WORDS;
        uint32_t word = load_word_clamped(in, words_per_row, W, H, origin_x + col * 4, origin_y + row);
        *reinterpret_cast<uint32_t*>(&tile[row][col * 4]) = word;
    }
    __syncthreads();

    const int x = (blockIdx.x * GRAY_BLOCK_X + threadIdx.x) * 4;
    const int y = blockIdx.y * GRAY_BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    uint32_t packed = 0;
    for (int k = 0; k < 4; ++k) {
        Window w{&tile[threadIdx.y + R][threadIdx.x * 4 + 4 + k], TILE_STRIDE, 1};
        packed |= static_cast<uint32_t>(Op::apply(w)) << (8 * k);
    }
    out[y * words_per_row + x / 4] = packed;
}

template <class Op>
__global__ void filter_rgba(const uint32_t* in, uint32_t* out, int W, int H) {
    constexpr int R = Op::radius;
    constexpr int TILE_H = RGBA_BLOCK_Y + 2 * R;
    constexpr int TILE_W = RGBA_BLOCK_X + 2 * R;

    __shared__ uint32_t tile[TILE_H][TILE_W];

    const int origin_x = blockIdx.x * RGBA_BLOCK_X - R;
    const int origin_y = blockIdx.y * RGBA_BLOCK_Y - R;
    const int tid = threadIdx.y * RGBA_BLOCK_X + threadIdx.x;

    for (int i = tid; i < TILE_H * TILE_W; i += RGBA_BLOCK_X * RGBA_BLOCK_Y) {
        int row = i / TILE_W;
        int col = i % TILE_W;
        int gx = min(max(origin_x + col, 0), W - 1);
        int gy = min(max(origin_y + row, 0), H - 1);
        tile[row][col] = in[gy * W + gx];
    }
    __syncthreads();

    const int x = blockIdx.x * RGBA_BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * RGBA_BLOCK_Y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    const uint8_t* base = reinterpret_cast<const uint8_t*>(&tile[threadIdx.y + R][threadIdx.x + R]);
    uint32_t packed = 0xFF000000u;
    for (int c = 0; c < 3; ++c) {
        Window w{base + c, TILE_W * 4, 4};
        packed |= static_cast<uint32_t>(Op::apply(w)) << (8 * c);
    }
    out[y * W + x] = packed;
}

static GaussWeights make_gauss_weights(float sigma) {
    std::array<double, GAUSS_SIZE * GAUSS_SIZE> raw{};
    double sum = 0.0;
    for (int dy = -GAUSS_RADIUS; dy <= GAUSS_RADIUS; ++dy) {
        for (int dx = -GAUSS_RADIUS; dx <= GAUSS_RADIUS; ++dx) {
            double v = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            raw[(dy + GAUSS_RADIUS) * GAUSS_SIZE + (dx + GAUSS_RADIUS)] = v;
            sum += v;
        }
    }

    GaussWeights weights{};
    for (size_t i = 0; i < raw.size(); ++i) {
        weights[i] = static_cast<float>(raw[i] / sum);
    }
    return weights;
}

static int clamped_pixel(const Plane& p, int W, int H, int x, int y) {
    x = std::min(std::max(x, 0), W - 1);
    y = std::min(std::max(y, 0), H - 1);
    return p[y * W + x];
}

static Plane cpu_gauss(const Plane& in, int W, int H, const GaussWeights& weights) {
    Plane out(in.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int dy = -GAUSS_RADIUS; dy <= GAUSS_RADIUS; ++dy) {
                for (int dx = -GAUSS_RADIUS; dx <= GAUSS_RADIUS; ++dx) {
                    sum += weights[(dy + GAUSS_RADIUS) * GAUSS_SIZE + (dx + GAUSS_RADIUS)]
                    * clamped_pixel(in, W, H, x + dx, y + dy);
                }
            }
            out[y * W + x] = clamp_u8(static_cast<int>(sum + 0.5f));
        }
    }
    return out;
}

static Plane cpu_emboss(const Plane& in, int W, int H) {
    Plane out(in.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sum = EMBOSS_BIAS;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    sum += EMBOSS_KERNEL[(dy + 1) * EMBOSS_SIZE + (dx + 1)]
                    * clamped_pixel(in, W, H, x + dx, y + dy);
                }
            }
            out[y * W + x] = clamp_u8(sum);
        }
    }
    return out;
}

static std::vector<Plane> split_planes(const std::vector<uint8_t>& data, int W, int H,
                                    const Layout& layout) {
    std::vector<Plane> planes(layout.channels, Plane(static_cast<size_t>(W) * H));
    for (int c = 0; c < layout.channels; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                planes[c][y * W + x] = data[y * layout.pitch + x * layout.pixel_stride + c];
            }
        }
    }
    return planes;
}

static bool matches(const std::vector<Plane>& reference, const std::vector<uint8_t>& data,
                    int W, int H, const Layout& layout) {
    for (int c = 0; c < layout.channels; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int got = data[y * layout.pitch + x * layout.pixel_stride + c];
                if (std::abs(got - static_cast<int>(reference[c][y * W + x])) > MAX_ALLOWED_DIFF) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <class Work>
static double time_cpu_ms(Work work) {
    auto t_start = std::chrono::high_resolution_clock::now();
    work();
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

template <class Launch>
static double time_gpu_ms(Launch launch) {
    launch();
    check(cudaDeviceSynchronize());

    cudaEvent_t start_event, stop_event;
    check(cudaEventCreate(&start_event));
    check(cudaEventCreate(&stop_event));

    check(cudaEventRecord(start_event, 0));
    launch();
    check(cudaEventRecord(stop_event, 0));
    check(cudaEventSynchronize(stop_event));

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start_event, stop_event));
    check(cudaEventDestroy(start_event));
    check(cudaEventDestroy(stop_event));
    return static_cast<double>(ms);
}

template <class Launch>
static bool measure(const char* name, int W, int H, const Layout& layout,
                    const uint32_t* d_in, const uint32_t* d_out,
                    const GaussWeights& weights, Launch launch) {
    const size_t bytes = static_cast<size_t>(layout.pitch) * H;
    std::vector<uint8_t> h_in(bytes);
    std::vector<uint8_t> h_out(bytes);
    check(cudaMemcpy(h_in.data(), d_in, bytes, cudaMemcpyDeviceToHost));

    const std::vector<Plane> source = split_planes(h_in, W, H, layout);
    std::vector<Plane> reference(source.size());
    double cpu_ms = time_cpu_ms([&] {
        for (size_t c = 0; c < source.size(); ++c) {
            reference[c] = cpu_emboss(cpu_gauss(source[c], W, H, weights), W, H);
        }
    });

    double gpu_ms = time_gpu_ms(launch);

    check(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool match = matches(reference, h_out, W, H, layout);

    std::cout << name << ": CPU " << cpu_ms << " ms, GPU " << gpu_ms << " ms, "
    << (match ? "MATCH" : "MISMATCH") << "\n";
    return match;
}

static bool run_gray(int W, int H, const GaussWeights& weights) {
    const int words_per_row = div_up(W, 4);
    const Layout layout{words_per_row * 4, 1, 1};
    const size_t bytes = static_cast<size_t>(layout.pitch) * H;

    DeviceBuffer in(bytes), tmp(bytes), out(bytes);
    dim3 block(GRAY_BLOCK_X, GRAY_BLOCK_Y);
    dim3 grid(div_up(words_per_row, GRAY_BLOCK_X), div_up(H, GRAY_BLOCK_Y));

    generate_gray<<<grid, block>>>(in.get(), W, H, words_per_row);
    check(cudaGetLastError());

    return measure("Gray ", W, H, layout, in.get(), out.get(), weights, [&] {
        filter_gray<GaussOp><<<grid, block>>>(in.get(), tmp.get(), W, H, words_per_row);
        filter_gray<EmbossOp><<<grid, block>>>(tmp.get(), out.get(), W, H, words_per_row);
        check(cudaGetLastError());
    });
}

static bool run_rgba(int W, int H, const GaussWeights& weights) {
    const Layout layout{W * 4, 4, 3};
    const size_t bytes = static_cast<size_t>(layout.pitch) * H;

    DeviceBuffer in(bytes), tmp(bytes), out(bytes);
    dim3 block(RGBA_BLOCK_X, RGBA_BLOCK_Y);
    dim3 grid(div_up(W, RGBA_BLOCK_X), div_up(H, RGBA_BLOCK_Y));

    generate_rgba<<<grid, block>>>(in.get(), W, H);
    check(cudaGetLastError());

    return measure("Color", W, H, layout, in.get(), out.get(), weights, [&] {
        filter_rgba<GaussOp><<<grid, block>>>(in.get(), tmp.get(), W, H);
        filter_rgba<EmbossOp><<<grid, block>>>(tmp.get(), out.get(), W, H);
        check(cudaGetLastError());
    });
}

static void parse_args(int argc, char* argv[], int& W, int& H) {
    W = 1920;
    H = 1080;
    if (argc >= 2) {
        W = std::stoi(argv[1]);
        H = W;
    }
    if (argc >= 3) {
        H = std::stoi(argv[2]);
    }
    if (W < DIM_MIN || W > DIM_MAX || H < DIM_MIN || H > DIM_MAX) {
        std::cerr << "W and H must be in [" << DIM_MIN << ", " << DIM_MAX << "]\n";
        std::exit(1);
    }
}

int main(int argc, char* argv[]) {
    int W = 0;
    int H = 0;
    parse_args(argc, argv, W, H);

    std::cout << "Lab 05, task 9: Gaussian 5x5 (sigma=1.0) -> Emboss 3x3\n";
    std::cout << "Image: " << W << "x" << H << "\n\n";

    GaussWeights weights = make_gauss_weights(GAUSS_SIGMA);
    check(cudaMemcpyToSymbol(c_gauss, weights.data(), sizeof(float) * weights.size()));
    check(cudaMemcpyToSymbol(c_emboss, EMBOSS_KERNEL.data(), sizeof(int) * EMBOSS_KERNEL.size()));

    std::cout << std::fixed << std::setprecision(2);

    bool gray_ok = run_gray(W, H, weights);
    bool rgba_ok = run_rgba(W, H, weights);

    return (gray_ok && rgba_ok) ? 0 : 1;
}
