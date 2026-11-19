#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

namespace fs = std::filesystem;

constexpr const char* DEFAULT_INPUT_DIR = "images/input";
constexpr const char* DEFAULT_OUTPUT_DIR = "images/output";
constexpr const char* FILTER_TAG = "gauss_emboss";

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

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

static int div_up(int a, int b) {
    return (a + b - 1) / b;
}

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

__global__ void rgba_to_gray(const uint32_t* rgba, uint32_t* gray, int W, int H, int words_per_row) {
    int word_x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (word_x >= words_per_row || y >= H) {
        return;
    }

    uint32_t word = 0;
    for (int k = 0; k < 4; ++k) {
        int x = word_x * 4 + k;
        if (x < W) {
            uint32_t p = rgba[y * W + x];
            int r = p & 0xFF;
            int g = (p >> 8) & 0xFF;
            int b = (p >> 16) & 0xFF;
            int luma = (77 * r + 150 * g + 29 * b + 128) >> 8;
            word |= static_cast<uint32_t>(luma) << (8 * k);
        }
    }
    gray[y * words_per_row + word_x] = word;
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

                                             static Plane cpu_pipeline(const Plane& in, int W, int H, const GaussWeights& weights) {
                                                 return cpu_emboss(cpu_gauss(in, W, H, weights), W, H);
                                             }

                                             static Plane extract_plane(const std::vector<uint8_t>& data, int W, int H,
                                                                        int pitch, int pixel_stride, int channel) {
                                                 Plane plane(static_cast<size_t>(W) * H);
                                                 for (int y = 0; y < H; ++y) {
                                                     for (int x = 0; x < W; ++x) {
                                                         plane[y * W + x] = data[y * pitch + x * pixel_stride + channel];
                                                     }
                                                 }
                                                 return plane;
                                                                        }

                                                                        static int max_diff(const Plane& ref, const std::vector<uint8_t>& data, int W, int H,
                                                                                            int pitch, int pixel_stride, int channel) {
                                                                            int worst = 0;
                                                                            for (int y = 0; y < H; ++y) {
                                                                                for (int x = 0; x < W; ++x) {
                                                                                    int got = data[y * pitch + x * pixel_stride + channel];
                                                                                    worst = std::max(worst, std::abs(got - static_cast<int>(ref[y * W + x])));
                                                                                }
                                                                            }
                                                                            return worst;
                                                                                            }

                                                                                            static std::vector<uint8_t> merge_rgba(const std::array<Plane, 3>& planes, int W, int H) {
                                                                                                const size_t pixels = static_cast<size_t>(W) * H;
                                                                                                std::vector<uint8_t> rgba(pixels * 4);
                                                                                                for (size_t i = 0; i < pixels; ++i) {
                                                                                                    for (int c = 0; c < 3; ++c) {
                                                                                                        rgba[i * 4 + c] = planes[c][i];
                                                                                                    }
                                                                                                    rgba[i * 4 + 3] = 255;
                                                                                                }
                                                                                                return rgba;
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

                                                                                            static void print_result(const char* type, double cpu_ms, double gpu_ms, bool match) {
                                                                                                std::cout << "  " << type << ": CPU " << cpu_ms << " ms, GPU " << gpu_ms << " ms, "
                                                                                                << (match ? "MATCH" : "MISMATCH") << "\n";
                                                                                            }

                                                                                            static void save_png(const std::string& path, int W, int H, int channels,
                                                                                                                 const uint8_t* data, int stride) {
                                                                                                if (!stbi_write_png(path.c_str(), W, H, channels, data, stride)) {
                                                                                                    std::cerr << "Failed to write " << path << "\n";
                                                                                                }
                                                                                                                 }

                                                                                                                 static std::string output_path(const fs::path& dir, const std::string& stem,
                                                                                                                                                const std::string& type, const std::string& method) {
                                                                                                                     return (dir / (stem + "_" + type + "_" + FILTER_TAG + "_" + method + ".png")).string();
                                                                                                                                                }

                                                                                                                                                static bool run_gray(const uint32_t* d_rgba, int W, int H, const std::string& stem,
                                                                                                                                                                     const fs::path& out_dir, const GaussWeights& weights) {
                                                                                                                                                    const int words_per_row = div_up(W, 4);
                                                                                                                                                    const int pitch = words_per_row * 4;
                                                                                                                                                    const size_t bytes = static_cast<size_t>(pitch) * H;

                                                                                                                                                    uint32_t* d_in = nullptr;
                                                                                                                                                    uint32_t* d_tmp = nullptr;
                                                                                                                                                    uint32_t* d_out = nullptr;
                                                                                                                                                    check(cudaMalloc(&d_in, bytes));
                                                                                                                                                    check(cudaMalloc(&d_tmp, bytes));
                                                                                                                                                    check(cudaMalloc(&d_out, bytes));

                                                                                                                                                    dim3 block(GRAY_BLOCK_X, GRAY_BLOCK_Y);
                                                                                                                                                    dim3 grid(div_up(words_per_row, GRAY_BLOCK_X), div_up(H, GRAY_BLOCK_Y));

                                                                                                                                                    rgba_to_gray<<<grid, block>>>(d_rgba, d_in, W, H, words_per_row);
                                                                                                                                                    check(cudaGetLastError());

                                                                                                                                                    std::vector<uint8_t> h_in(bytes);
                                                                                                                                                    std::vector<uint8_t> h_out(bytes);
                                                                                                                                                    check(cudaMemcpy(h_in.data(), d_in, bytes, cudaMemcpyDeviceToHost));

                                                                                                                                                    const Plane source = extract_plane(h_in, W, H, pitch, 1, 0);
                                                                                                                                                    Plane reference;
                                                                                                                                                    double cpu_ms = time_cpu_ms([&] {
                                                                                                                                                        reference = cpu_pipeline(source, W, H, weights);
                                                                                                                                                    });

                                                                                                                                                    double gpu_ms = time_gpu_ms([&] {
                                                                                                                                                        filter_gray<GaussOp><<<grid, block>>>(d_in, d_tmp, W, H, words_per_row);
                                                                                                                                                        filter_gray<EmbossOp><<<grid, block>>>(d_tmp, d_out, W, H, words_per_row);
                                                                                                                                                        check(cudaGetLastError());
                                                                                                                                                    });

                                                                                                                                                    check(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
                                                                                                                                                    bool match = max_diff(reference, h_out, W, H, pitch, 1, 0) <= MAX_ALLOWED_DIFF;

                                                                                                                                                    save_png((out_dir / (stem + "_gray_input.png")).string(), W, H, 1, h_in.data(), pitch);
                                                                                                                                                    save_png(output_path(out_dir, stem, "gray", "cpu"), W, H, 1, reference.data(), W);
                                                                                                                                                    save_png(output_path(out_dir, stem, "gray", "gpu"), W, H, 1, h_out.data(), pitch);

                                                                                                                                                    check(cudaFree(d_in));
                                                                                                                                                    check(cudaFree(d_tmp));
                                                                                                                                                    check(cudaFree(d_out));

                                                                                                                                                    print_result("gray ", cpu_ms, gpu_ms, match);
                                                                                                                                                    return match;
                                                                                                                                                                     }

                                                                                                                                                                     static bool run_rgba(const uint32_t* d_rgba, const std::vector<uint8_t>& h_rgba, int W, int H,
                                                                                                                                                                                          const std::string& stem, const fs::path& out_dir, const GaussWeights& weights) {
                                                                                                                                                                         const int pitch = W * 4;
                                                                                                                                                                         const size_t bytes = static_cast<size_t>(pitch) * H;

                                                                                                                                                                         uint32_t* d_tmp = nullptr;
                                                                                                                                                                         uint32_t* d_out = nullptr;
                                                                                                                                                                         check(cudaMalloc(&d_tmp, bytes));
                                                                                                                                                                         check(cudaMalloc(&d_out, bytes));

                                                                                                                                                                         dim3 block(RGBA_BLOCK_X, RGBA_BLOCK_Y);
                                                                                                                                                                         dim3 grid(div_up(W, RGBA_BLOCK_X), div_up(H, RGBA_BLOCK_Y));

                                                                                                                                                                         std::array<Plane, 3> source;
                                                                                                                                                                         std::array<Plane, 3> reference;
                                                                                                                                                                         for (int c = 0; c < 3; ++c) {
                                                                                                                                                                             source[c] = extract_plane(h_rgba, W, H, pitch, 4, c);
                                                                                                                                                                         }

                                                                                                                                                                         double cpu_ms = time_cpu_ms([&] {
                                                                                                                                                                             for (int c = 0; c < 3; ++c) {
                                                                                                                                                                                 reference[c] = cpu_pipeline(source[c], W, H, weights);
                                                                                                                                                                             }
                                                                                                                                                                         });

                                                                                                                                                                         double gpu_ms = time_gpu_ms([&] {
                                                                                                                                                                             filter_rgba<GaussOp><<<grid, block>>>(d_rgba, d_tmp, W, H);
                                                                                                                                                                             filter_rgba<EmbossOp><<<grid, block>>>(d_tmp, d_out, W, H);
                                                                                                                                                                             check(cudaGetLastError());
                                                                                                                                                                         });

                                                                                                                                                                         std::vector<uint8_t> h_out(bytes);
                                                                                                                                                                         check(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

                                                                                                                                                                         int diff = 0;
                                                                                                                                                                         for (int c = 0; c < 3; ++c) {
                                                                                                                                                                             diff = std::max(diff, max_diff(reference[c], h_out, W, H, pitch, 4, c));
                                                                                                                                                                         }
                                                                                                                                                                         bool match = diff <= MAX_ALLOWED_DIFF;

                                                                                                                                                                         const std::vector<uint8_t> cpu_rgba = merge_rgba(reference, W, H);
                                                                                                                                                                         save_png(output_path(out_dir, stem, "color", "cpu"), W, H, 4, cpu_rgba.data(), pitch);
                                                                                                                                                                         save_png(output_path(out_dir, stem, "color", "gpu"), W, H, 4, h_out.data(), pitch);

                                                                                                                                                                         check(cudaFree(d_tmp));
                                                                                                                                                                         check(cudaFree(d_out));

                                                                                                                                                                         print_result("color", cpu_ms, gpu_ms, match);
                                                                                                                                                                         return match;
                                                                                                                                                                                          }

                                                                                                                                                                                          static bool process_image(const fs::path& path, const fs::path& out_dir, const GaussWeights& weights) {
                                                                                                                                                                                              int W = 0;
                                                                                                                                                                                              int H = 0;
                                                                                                                                                                                              int channels = 0;
                                                                                                                                                                                              uint8_t* pixels = stbi_load(path.string().c_str(), &W, &H, &channels, 4);
                                                                                                                                                                                              if (pixels == nullptr) {
                                                                                                                                                                                                  std::cout << path.filename().string() << ": skipped (cannot read)\n\n";
                                                                                                                                                                                                  return true;
                                                                                                                                                                                              }
                                                                                                                                                                                              if (W < DIM_MIN || W > DIM_MAX || H < DIM_MIN || H > DIM_MAX) {
                                                                                                                                                                                                  stbi_image_free(pixels);
                                                                                                                                                                                                  std::cout << path.filename().string() << ": skipped (size " << W << "x" << H
                                                                                                                                                                                                  << " is outside [" << DIM_MIN << ", " << DIM_MAX << "])\n\n";
                                                                                                                                                                                                  return true;
                                                                                                                                                                                              }

                                                                                                                                                                                              const size_t bytes = static_cast<size_t>(W) * H * 4;
                                                                                                                                                                                              std::vector<uint8_t> h_rgba(pixels, pixels + bytes);
                                                                                                                                                                                              stbi_image_free(pixels);

                                                                                                                                                                                              uint32_t* d_rgba = nullptr;
                                                                                                                                                                                              check(cudaMalloc(&d_rgba, bytes));
                                                                                                                                                                                              check(cudaMemcpy(d_rgba, h_rgba.data(), bytes, cudaMemcpyHostToDevice));

                                                                                                                                                                                              std::cout << path.filename().string() << " (" << W << "x" << H << ")\n";
                                                                                                                                                                                              const std::string stem = path.stem().string();
                                                                                                                                                                                              bool gray_ok = run_gray(d_rgba, W, H, stem, out_dir, weights);
                                                                                                                                                                                              bool color_ok = run_rgba(d_rgba, h_rgba, W, H, stem, out_dir, weights);
                                                                                                                                                                                              std::cout << "\n";

                                                                                                                                                                                              check(cudaFree(d_rgba));
                                                                                                                                                                                              return gray_ok && color_ok;
                                                                                                                                                                                          }

                                                                                                                                                                                          static bool is_supported_image(const fs::path& path) {
                                                                                                                                                                                              std::string ext = path.extension().string();
                                                                                                                                                                                              std::transform(ext.begin(), ext.end(), ext.begin(),
                                                                                                                                                                                                             [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                                                                                                                                                                                              return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
                                                                                                                                                                                          }

                                                                                                                                                                                          static std::vector<fs::path> list_images(const fs::path& dir) {
                                                                                                                                                                                              std::vector<fs::path> images;
                                                                                                                                                                                              if (!fs::is_directory(dir)) {
                                                                                                                                                                                                  return images;
                                                                                                                                                                                              }
                                                                                                                                                                                              for (const auto& entry : fs::directory_iterator(dir)) {
                                                                                                                                                                                                  if (entry.is_regular_file() && is_supported_image(entry.path())) {
                                                                                                                                                                                                      images.push_back(entry.path());
                                                                                                                                                                                                  }
                                                                                                                                                                                              }
                                                                                                                                                                                              std::sort(images.begin(), images.end());
                                                                                                                                                                                              return images;
                                                                                                                                                                                          }

                                                                                                                                                                                          int main(int argc, char* argv[]) {
                                                                                                                                                                                              const fs::path in_dir = argc >= 2 ? argv[1] : DEFAULT_INPUT_DIR;
                                                                                                                                                                                              const fs::path out_dir = argc >= 3 ? argv[2] : DEFAULT_OUTPUT_DIR;

                                                                                                                                                                                              const std::vector<fs::path> images = list_images(in_dir);
                                                                                                                                                                                              if (images.empty()) {
                                                                                                                                                                                                  std::cerr << "No images found in " << in_dir.string() << "\n";
                                                                                                                                                                                                  return 1;
                                                                                                                                                                                              }
                                                                                                                                                                                              fs::create_directories(out_dir);

                                                                                                                                                                                              GaussWeights weights = make_gauss_weights(GAUSS_SIGMA);
                                                                                                                                                                                              check(cudaMemcpyToSymbol(c_gauss, weights.data(), sizeof(float) * weights.size()));
                                                                                                                                                                                              check(cudaMemcpyToSymbol(c_emboss, EMBOSS_KERNEL.data(), sizeof(int) * EMBOSS_KERNEL.size()));

                                                                                                                                                                                              std::cout << "Lab 05, task 9: Gaussian 5x5 (sigma=1.0) -> Emboss 3x3\n\n";
                                                                                                                                                                                              std::cout << std::fixed << std::setprecision(2);

                                                                                                                                                                                              bool all_ok = true;
                                                                                                                                                                                              for (const fs::path& image : images) {
                                                                                                                                                                                                  all_ok = process_image(image, out_dir, weights) && all_ok;
                                                                                                                                                                                              }

                                                                                                                                                                                              return all_ok ? 0 : 1;
                                                                                                                                                                                          }
