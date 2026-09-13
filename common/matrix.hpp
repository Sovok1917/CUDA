#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <new>

/**
 * Purpose: Represents an inner 4x4 submatrix of 32-bit floating point numbers.
 *          16-byte aligned for direct compatibility with SSE 128-bit operations.
 */
struct alignas(16) Block4x4 {
    float data[4][4];
};

/**
 * Purpose: Block matrix container of dimensions [Rows x Cols], where each element
 *          is an inner Block4x4. Memory is laid out contiguously.
 */
class BlockMatrix {
public:
    /**
     * Purpose: Constructs an uninitialized block matrix.
     * Input:   rows - number of outer block rows.
     *          cols - number of outer block columns.
     */
    BlockMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), blocks_(rows * cols) {}

    /**
     * Purpose: Accesses block at outer coordinate (row, col).
     * Input:   row - outer row index.
     *          col - outer column index.
     * Return:  Block4x4& - reference to block.
     */
    Block4x4& at(size_t row, size_t col) {
        return blocks_[row * cols_ + col];
    }

    /**
     * Purpose: Const access to block at outer coordinate (row, col).
     * Input:   row - outer row index.
     *          col - outer column index.
     * Return:  const Block4x4& - const reference to block.
     */
    const Block4x4& at(size_t row, size_t col) const {
        return blocks_[row * cols_ + col];
    }

    /**
     * Purpose: Clears all blocks in the matrix to zero.
     * Input:   None.
     * Return:  void.
     */
    void zero() {
        for (auto& block : blocks_) {
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    block.data[r][c] = 0.0f;
                }
            }
        }
    }

    /**
     * Purpose: Populates the matrix with pseudo-random float values.
     * Input:   min_val - minimum generated value.
     *          max_val - maximum generated value.
     *          seed    - random generator seed.
     * Return:  void.
     */
    void randomize(float min_val = -1.0f, float max_val = 1.0f, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (auto& block : blocks_) {
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    block.data[r][c] = dist(gen);
                }
            }
        }
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const Block4x4* data() const { return blocks_.data(); }
    Block4x4* data() { return blocks_.data(); }

private:
    size_t rows_;
    size_t cols_;
    std::vector<Block4x4> blocks_;
};

/**
 * Purpose: Compares two matrices element-by-element within a given relative tolerance.
 * Input:   a       - first matrix.
 *          b       - second matrix.
 *          epsilon - allowed difference tolerance.
 * Return:  bool - true if matrices match, false otherwise.
 */
inline bool verify_matrices(const BlockMatrix& a, const BlockMatrix& b, float epsilon = 1e-4f) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }

    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            const auto& block_a = a.at(i, j);
            const auto& block_b = b.at(i, j);

            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    float diff = std::fabs(block_a.data[r][c] - block_b.data[r][c]);
                    if (diff > epsilon) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
