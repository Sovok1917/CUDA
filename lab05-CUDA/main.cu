#include <algorithm>
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
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int DIM_MIN = 10;
constexpr int DIM_MAX = 8192;
constexpr int BLOCK = 16;
constexpr int GAUSS_RADIUS = 2;
constexpr int EMBOSS_RADIUS = 1;
constexpr int MAX_TILE = BLOCK + 2 * GAUSS_RADIUS;
constexpr float GAUSS_SIGMA = 1.0f;
constexpr float EMBOSS_BIAS = 128.0f;
constexpr int MAX_ALLOWED_DIFF = 1;
const std::string INPUT_DIR = "images/input";
const std::string OUTPUT_DIR = "images/output";

__constant__ float c_gauss[25];
__constant__ float c_emboss[9];

struct Image {
    int W = 0;
    int H = 0;
    int channels = 0;
    std::vector<uint8_t> pixels;
};

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

__global__ void convolve_gray(const uint8_t* src, uint8_t* dst, int W, int H,
                              int R, float bias, bool use_emboss) {
    const int K = 2 * R + 1;
    const int tile_size = BLOCK + 2 * R;
    __shared__ uint8_t tile[MAX_TILE][MAX_TILE];

    for (int ty = threadIdx.y; ty < tile_size; ty += BLOCK) {
        for (int tx = threadIdx.x; tx < tile_size; tx += BLOCK) {
            int gx = min(max(static_cast<int>(blockIdx.x) * BLOCK + tx - R, 0), W - 1);
            int gy = min(max(static_cast<int>(blockIdx.y) * BLOCK + ty - R, 0), H - 1);
            tile[ty][tx] = src[gy * W + gx];
        }
    }
    __syncthreads();

    int x = static_cast<int>(blockIdx.x) * BLOCK + threadIdx.x;
    int y = static_cast<int>(blockIdx.y) * BLOCK + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    float sum = bias;
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            float w = use_emboss ? c_emboss[dy * K + dx] : c_gauss[dy * K + dx];
            sum += w * tile[threadIdx.y + dy][threadIdx.x + dx];
        }
    }
    dst[y * W + x] = to_u8(sum);
}

__global__ void convolve_rgb(const uint8_t* src_bytes, uint8_t* dst_bytes, int W, int H,
                             int R, float bias, bool use_emboss) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(src_bytes);
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst_bytes);
    const int K = 2 * R + 1;
    const int tile_size = BLOCK + 2 * R;
    __shared__ uint32_t tile[MAX_TILE][MAX_TILE];

    for (int ty = threadIdx.y; ty < tile_size; ty += BLOCK) {
        for (int tx = threadIdx.x; tx < tile_size; tx += BLOCK) {
            int gx = min(max(static_cast<int>(blockIdx.x) * BLOCK + tx - R, 0), W - 1);
            int gy = min(max(static_cast<int>(blockIdx.y) * BLOCK + ty - R, 0), H - 1);
            tile[ty][tx] = src[gy * W + gx];
        }
    }
    __syncthreads();

    int x = static_cast<int>(blockIdx.x) * BLOCK + threadIdx.x;
    int y = static_cast<int>(blockIdx.y) * BLOCK + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }

    float r = bias;
    float g = bias;
    float b = bias;
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            float w = use_emboss ? c_emboss[dy * K + dx] : c_gauss[dy * K + dx];
            uint32_t p = tile[threadIdx.y + dy][threadIdx.x + dx];
            r += w * (p & 0xFFu);
            g += w * ((p >> 8) & 0xFFu);
            b += w * ((p >> 16) & 0xFFu);
        }
    }
    dst[y * W + x] = to_u8(r) | (to_u8(g) << 8) | (to_u8(b) << 16);
}

void convolve(const Image& img, const uint8_t* src, uint8_t* dst,
              int R, float bias, bool use_emboss) {
    const dim3 block(BLOCK, BLOCK);
    const dim3 grid((img.W + BLOCK - 1) / BLOCK, (img.H + BLOCK - 1) / BLOCK);
    if (img.channels == 3) {
        convolve_rgb<<<grid, block>>>(src, dst, img.W, img.H, R, bias, use_emboss);
    } else {
        convolve_gray<<<grid, block>>>(src, dst, img.W, img.H, R, bias, use_emboss);
    }
    check(cudaGetLastError());
}

void gpu_pipeline(const Image& img, uint8_t* in, uint8_t* tmp, uint8_t* out) {
    convolve(img, in, tmp, GAUSS_RADIUS, 0.0f, false);
    convolve(img, tmp, out, EMBOSS_RADIUS, EMBOSS_BIAS, true);
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

Image load_image(const std::string& path, int channels) {
    Image img;
    img.channels = channels;
    int file_channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &img.W, &img.H, &file_channels, channels);
    if (!data) {
        std::cerr << "Cannot load image: " << path << "\n";
        std::exit(1);
    }
    img.pixels.assign(data, data + static_cast<size_t>(img.W) * img.H * channels);
    stbi_image_free(data);
    return img;
}

void save_png(const std::string& path, const Image& img, const std::vector<uint8_t>& pixels) {
    if (!stbi_write_png(path.c_str(), img.W, img.H, img.channels, pixels.data(), img.W * img.channels)) {
        std::cerr << "Cannot write image: " << path << "\n";
        std::exit(1);
    }
}

std::vector<uint8_t> to_device(const Image& img) {
    if (img.channels == 1) {
        return img.pixels;
    }
    const size_t count = static_cast<size_t>(img.W) * img.H;
    std::vector<uint8_t> packed(count * sizeof(uint32_t));
    uint32_t* words = reinterpret_cast<uint32_t*>(packed.data());
    for (size_t i = 0; i < count; ++i) {
        words[i] = img.pixels[3 * i] | (img.pixels[3 * i + 1] << 8) | (img.pixels[3 * i + 2] << 16);
    }
    return packed;
}

std::vector<uint8_t> from_device(const Image& img, const std::vector<uint8_t>& device_bytes) {
    if (img.channels == 1) {
        return device_bytes;
    }
    const size_t count = static_cast<size_t>(img.W) * img.H;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(device_bytes.data());
    std::vector<uint8_t> rgb(count * 3);
    for (size_t i = 0; i < count; ++i) {
        rgb[3 * i] = static_cast<uint8_t>(words[i] & 0xFFu);
        rgb[3 * i + 1] = static_cast<uint8_t>((words[i] >> 8) & 0xFFu);
        rgb[3 * i + 2] = static_cast<uint8_t>((words[i] >> 16) & 0xFFu);
    }
    return rgb;
}

std::vector<uint8_t> cpu_convolve(const Image& img, const std::vector<uint8_t>& src,
                                  const std::vector<float>& weights, int R, float bias) {
    const int K = 2 * R + 1;
    const int C = img.channels;
    std::vector<uint8_t> dst(src.size());
    for (int y = 0; y < img.H; ++y) {
        for (int x = 0; x < img.W; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = bias;
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        int sx = std::min(std::max(x + dx, 0), img.W - 1);
                        int sy = std::min(std::max(y + dy, 0), img.H - 1);
                        sum += weights[(dy + R) * K + (dx + R)] * src[(sy * img.W + sx) * C + c];
                    }
                }
                dst[(y * img.W + x) * C + c] = to_u8(sum);
            }
        }
    }
    return dst;
}

bool run(const Image& img, const char* name, const std::string& output_base,
         const std::vector<float>& gauss, const std::vector<float>& emboss) {
    const auto cpu_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> blurred = cpu_convolve(img, img.pixels, gauss, GAUSS_RADIUS, 0.0f);
    const std::vector<uint8_t> reference = cpu_convolve(img, blurred, emboss, EMBOSS_RADIUS, EMBOSS_BIAS);
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    const std::vector<uint8_t> device_in = to_device(img);
    const size_t bytes = device_in.size();
    uint8_t* d_in;
    uint8_t* d_tmp;
    uint8_t* d_out;
    check(cudaMalloc(&d_in, bytes));
    check(cudaMalloc(&d_tmp, bytes));
    check(cudaMalloc(&d_out, bytes));
    check(cudaMemcpy(d_in, device_in.data(), bytes, cudaMemcpyHostToDevice));

    gpu_pipeline(img, d_in, d_tmp, d_out);
    check(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    gpu_pipeline(img, d_in, d_tmp, d_out);
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    check(cudaEventElapsedTime(&gpu_ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    std::vector<uint8_t> device_mid(bytes);
    std::vector<uint8_t> device_out(bytes);
    check(cudaMemcpy(device_mid.data(), d_tmp, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(device_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(d_in));
    check(cudaFree(d_tmp));
    check(cudaFree(d_out));

    const std::vector<uint8_t> host_mid = from_device(img, device_mid);
    const std::vector<uint8_t> host_out = from_device(img, device_out);

    const std::string gauss_path = output_base + "_gauss.png";
    const std::string emboss_path = output_base + "_emboss.png";
    save_png(gauss_path, img, host_mid);
    save_png(emboss_path, img, host_out);

    int max_diff = 0;
    for (size_t i = 0; i < host_out.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(host_out[i]) - static_cast<int>(reference[i])));
    }
    const bool match = max_diff <= MAX_ALLOWED_DIFF;
    const double speedup = gpu_ms > 0.0f ? cpu_ms / gpu_ms : 0.0;
    std::cout << name << ": CPU " << cpu_ms << " ms, GPU " << gpu_ms << " ms, speedup "
              << speedup << "x, max diff " << max_diff << ", "
              << (match ? "MATCH" : "MISMATCH") << "\n";
    std::cout << "  -> " << gauss_path << "\n";
    std::cout << "  -> " << emboss_path << "\n";
    return match;
}

bool is_image(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tga";
}

bool process_image(const std::filesystem::path& path, const std::vector<float>& gauss,
                   const std::vector<float>& emboss) {
    const std::string input_path = path.string();
    const std::string stem = path.stem().string();

    const Image color = load_image(input_path, 3);
    if (color.W < DIM_MIN || color.W > DIM_MAX || color.H < DIM_MIN || color.H > DIM_MAX) {
        std::cerr << "Skipping " << input_path << ": W and H must be in ["
                  << DIM_MIN << ", " << DIM_MAX << "]\n";
        return false;
    }
    const Image gray = load_image(input_path, 1);

    std::cout << "Image: " << input_path << " (" << color.W << "x" << color.H << ")\n";
    const bool gray_ok = run(gray, "Gray ", OUTPUT_DIR + "/" + stem + "_gray", gauss, emboss);
    const bool color_ok = run(color, "Color", OUTPUT_DIR + "/" + stem + "_color", gauss, emboss);
    std::cout << "\n";
    return gray_ok && color_ok;
}

int main() {
    std::vector<std::filesystem::path> inputs;
    for (const auto& entry : std::filesystem::directory_iterator(INPUT_DIR)) {
        if (entry.is_regular_file() && is_image(entry.path())) {
            inputs.push_back(entry.path());
        }
    }
    if (inputs.empty()) {
        std::cerr << "No images found in " << INPUT_DIR << "\n";
        return 1;
    }
    std::sort(inputs.begin(), inputs.end());
    std::filesystem::create_directories(OUTPUT_DIR);

    const std::vector<float> gauss = make_gauss(GAUSS_SIGMA);
    const std::vector<float> emboss = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    check(cudaMemcpyToSymbol(c_gauss, gauss.data(), gauss.size() * sizeof(float)));
    check(cudaMemcpyToSymbol(c_emboss, emboss.data(), emboss.size() * sizeof(float)));

    std::cout << "Lab 05, task 9: Gaussian 5x5 (sigma=1.0) -> Emboss 3x3\n";
    std::cout << "Found " << inputs.size() << " image(s)\n\n";
    std::cout << std::fixed << std::setprecision(2);

    bool all_ok = true;
    for (const auto& path : inputs) {
        all_ok = process_image(path, gauss, emboss) && all_ok;
    }
    return all_ok ? 0 : 1;
}
