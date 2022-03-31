#ifndef PACKER_HPP
#define PACKER_HPP

#define DLL_EXPORT __declspec(dllexport)

#include "common.hpp"
#include <type_traits>


struct Board {
    //all pieces except kings
    uint8_t piece_sq[30];
    Piece piece[30];
    uint8_t kings[2];

    uint8_t num_pieces;
    Color stm;
};

//Assumes little-endianness
struct BitStream {
    uint8_t* bytes;
    size_t cursor;

    void write_bit(uint8_t bit);
    uint8_t read_bit();

    void write_piece(Piece p);
    Piece read_piece();

    template<typename T>
    void write_num(T x) {
        static_assert(std::is_integral_v<T>, "must be integral");
        constexpr int N = sizeof(T) * 8;
        
        for (int i = N - 1; i >= 0; --i)
            write_bit((x >> i) & 1);
    }

    template<typename T>
    T read_num() {
        static_assert(std::is_integral_v<T>, "must be integral");
        constexpr int N = sizeof(T) * 8;

        T x = read_bit();
        for (int i = 1; i < N; ++i) {
            x <<= 1;
            x |= read_bit();
        }

        return x;
    }
};

void encode_board(BitStream &bs, const char *fen);
Board decode_board(BitStream &bs);


extern "C" DLL_EXPORT int open_file(const char *path);
extern "C" DLL_EXPORT int close_file();
extern "C" DLL_EXPORT int write_entry(const char *fen, int score);

#endif
