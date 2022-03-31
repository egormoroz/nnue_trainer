#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <cassert>

#ifdef _MSC_VER
#pragma warning(disable:4146)
#include <intrin.h>
#endif

/*---------------Define some intrisincs--------------*/

#if defined(__GNUC__)  // GCC, Clang, ICC

constexpr uint8_t lsb(uint64_t b) {
    assert(b);
    return uint8_t(__builtin_ctzll(b));
}

#elif defined(_MSC_VER)  // MSVC

#ifdef _WIN64  // MSVC, WIN64

inline uint8_t lsb(uint64_t b) {
    assert(b);
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return (uint8_t)idx;
}

#endif // WIN64
#endif // MSVC

/*-------------End of intrisinc defitions------------*/

inline uint8_t pop_lsb(uint64_t& bb) {
    uint8_t sq = lsb(bb);
    bb &= bb - 1;
    return sq;
}

enum PieceType {
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    NONE,
};

enum Color : uint8_t {
    WHITE,
    BLACK,
};

enum Piece : uint8_t {
    W_PAWN, B_PAWN,
    W_KNIGHT, B_KNIGHT,
    W_BISHOP, B_BISHOP,
    W_ROOK, B_ROOK,
    W_QUEEN, B_QUEEN,
    W_KING, B_KING,
    W_NONE, B_NONE,
};

constexpr char piece_char[] = {
    'P', 'p',
    'N', 'n',
    'B', 'b',
    'R', 'r',
    'Q', 'q',
    'K', 'k',
    '?', '?',
};

constexpr PieceType type_of(Piece p) { return PieceType(p >> 1); }
constexpr Color color_of(Piece p) { return Color(p & 1); }

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece((pt << 1) + c);
}


constexpr bool is_upper(char ch) {
    return ch >= 'A' && ch <= 'Z';
}

constexpr bool is_lower(char ch) {
    return ch >= 'a' && ch <= 'z';
}

constexpr bool is_digit(char ch) {
    return ch >= '0' && ch <= '9';
}

constexpr Piece piece_from_ch(char ch) {
    Color c = WHITE;
    if (is_lower(ch)) {
        c = BLACK;
        ch -= ('a' - 'A');
    }

    switch (ch) {
    case 'P': return make_piece(c, PAWN);
    case 'N': return make_piece(c, KNIGHT);
    case 'B': return make_piece(c, BISHOP);
    case 'R': return make_piece(c, ROOK);
    case 'Q': return make_piece(c, QUEEN);
    case 'K': return make_piece(c, KING);
    default: return W_NONE;
    };
}


#endif
