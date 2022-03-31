#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <fstream>
#include "common.hpp"
#include "packer.hpp"

namespace huffman {

struct Encoding {
    uint8_t bits;
    uint8_t n_bits;
};

constexpr Encoding e_table[] = {
    { 0b0,    1 }, //pawn
    { 0b111,  3 }, //knight
    { 0b100,  3 }, //bishop
    { 0b1100, 4 }, //rook
    { 0b1101, 4 }, //queen
    { 0b101,  3 }, //king
};

constexpr PieceType d_table[] = {
    PAWN, //0
    NONE, NONE, NONE,
    BISHOP, //4
    KING, //5
    NONE,
    KNIGHT, //7
    NONE, NONE, NONE, NONE,
    ROOK, //12
    QUEEN, //13
    NONE,
    NONE
};

} //namespace huffman


void BitStream::write_bit(uint8_t bit) {
    size_t byte_idx = cursor >> 3,
        bit_idx = cursor & 7;
    bytes[byte_idx] &= ~(1 << bit_idx);
    bytes[byte_idx] |= bit << bit_idx;
    ++cursor;
}

uint8_t BitStream::read_bit() {
    size_t byte_idx = cursor >> 3,
        bit_idx = cursor & 7;
    ++cursor;
    return (bytes[byte_idx] >> bit_idx) & 1;
}

void BitStream::write_piece(Piece p) {
    write_bit(color_of(p));
    PieceType pt = type_of(p);
    for (int i = huffman::e_table[pt].n_bits - 1; i >= 0; --i)
        write_bit((huffman::e_table[pt].bits >> i) & 1);
}

Piece BitStream::read_piece() {
    Color c = Color(read_bit());
    uint8_t code = 0;
    do {
        code |= read_bit();

        if (huffman::d_table[code] != NONE)
            return make_piece(c, huffman::d_table[code]);

        code <<= 1;
    } while (code < 16);

    return W_NONE;
}

Board decode_board(BitStream& bs) {
    Board b;
    //this is suboptimal but I don't care
    b.stm = Color(bs.read_bit());
    b.num_pieces = 0;

    uint64_t mask = bs.read_num<uint64_t>();
    while (mask) {
        uint8_t sq = pop_lsb(mask);
        Piece p = bs.read_piece();
        assert(type_of(p) != NONE);

        if (type_of(p) == KING) {
            b.kings[color_of(p)] = sq;
            continue;
        }

        b.piece_sq[b.num_pieces] = sq;
        b.piece[b.num_pieces] = p;
        b.num_pieces++;
    }

    return b;
}

void encode_board(BitStream& bs, const char* fen) {
    Piece pieces[64];
    Color stm;

    memset(pieces, W_NONE, sizeof(pieces));

    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            char ch = *fen++;
            if (is_digit(ch)) {
                f += ch - '1';
                continue;
            }

            pieces[r * 8 + f] = piece_from_ch(ch);
        }
        ++fen;
    }

    stm = *fen == 'w' ? WHITE : BLACK;

    bs.write_bit(stm);
    for (int i = 63; i >= 0; --i) {
        uint8_t bit = type_of(pieces[i]) != NONE;
        bs.write_bit(bit);
    }

    for (int i = 0; i < 64; ++i) {
        if (type_of(pieces[i]) != NONE)
            bs.write_piece(pieces[i]);
    }
}

static std::ofstream fout;

static uint8_t buffer[64];
static BitStream bs{ (uint8_t*)&buffer, 0 };

int write_entry(const char *fen, int score) {
    if (!fout.is_open())
        return 0;

    bs.write_num(static_cast<int16_t>(score));
    encode_board(bs, fen);

    fout.write((const char*)buffer, bs.cursor / 8);
    buffer[0] = buffer[bs.cursor / 8];
    bs.cursor = bs.cursor % 8;
    return 1;
}

int open_file(const char *path)  {
    fout.open(path, std::ios::binary | std::ios::app);
    return fout.is_open();
}

int close_file() {
    if (bs.cursor && fout.is_open())
        fout.write((const char*)buffer, (bs.cursor + 7) / 8);
    bs.cursor = 0;
    if (fout.is_open()) {
        fout.close();
        return 1;
    }

    return 0;
}

BOOL WINAPI DllMain(
    HINSTANCE ,  // handle to DLL module
    DWORD ,     // reason for calling function
    LPVOID )  // reserved
{
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}

