#ifndef BITUTIL_HPP
#define BITUTIL_HPP

#include <cstdint>
#include <type_traits>

struct BitReader {
    const uint8_t *data{};
    size_t cursor{};

    uint8_t read_bit() {
        size_t byte_idx = cursor / 8, bit_idx = cursor % 8;
        ++cursor;
        return (data[byte_idx] >> bit_idx) & 1;
    }

    uint16_t read16() {
        size_t byte_idx = cursor / 8, bit_idx = cursor % 8;
        cursor += 16;
        uint32_t x = *(const uint32_t*)&data[byte_idx];
        x >>= bit_idx;
        return x & 0xFFFF;
    }

    uint64_t read64() {
        size_t byte_idx = cursor / 8, bit_idx = cursor % 8;
        cursor += 64;
        uint64_t x = data[byte_idx + 8];
        x <<= 63 - bit_idx;
        x <<= 1;
        x |= *(const uint64_t*)&data[byte_idx] >> bit_idx;
        return x;
    }
};

//assumes data is initialized with zeros
struct BitWriter {
    uint8_t *data{};
    size_t cursor{};

    void write_bit(uint8_t bit) {
        size_t byte_idx = cursor / 8, bit_idx = cursor % 8;
        ++cursor;
        data[byte_idx] |= (bit & 1) << bit_idx;
    }

    template<typename T>
    void write(T x, size_t n_bits = sizeof(T) * 8) {
        using U = std::decay_t<T>;
        static_assert(std::is_integral_v<U>, "must be integral");
        static_assert(!std::is_same_v<U, bool>, "use write_bit to write a single bit");

        x &= (2ull << (n_bits - 1)) - 1ull;

        size_t byte_idx = cursor / 8, bit_idx = cursor % 8;
        cursor += n_bits;
        data[byte_idx] |= static_cast<uint8_t>(x) << bit_idx;
        *(U*)&data[byte_idx + 1] |= x >> (8 - bit_idx);
    }
};

#endif
