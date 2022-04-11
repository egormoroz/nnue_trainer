#include "fen.hpp"
#include <cstring>

bool parse_fen(std::string_view fen, Board &b) {
    b.n_pieces = 0;
    b.mask = 0;
    Piece pieces[64];
    memset(pieces, W_NONE, sizeof(pieces));

    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f){ 
            if (fen.empty())
                return false;

            int s = r * 8 + f;
            char ch = fen.front();
            fen = fen.substr(1);
            if (is_digit(ch)) {
                f += ch - '1';
                continue;
            }

            Piece p = piece_from_ch(ch);
            if (type_of(p) == NONE)
                return false;

            b.mask |= 1ull << s;
            pieces[s] = p;
        }

        if (r != 0 && (fen.empty() || fen.front() != '/'))
            return false;
        fen = fen.substr(1);
    }

    if (fen.empty())
        return false;

    switch (fen.front()) {
    case 'w': b.stm = WHITE; break;
    case 'b': b.stm = BLACK; break;
    default: return false;
    };

    for (int i = 0; i < 64; ++i) {
        if (pieces[i] != W_NONE)
            b.pieces[b.n_pieces++] = pieces[i];
    }

    return true;
}

char* get_fen(const TrainingEntry &e, char *buffer) {
    Piece pieces[64];
    memset(pieces, W_NONE, sizeof(pieces));
    for (int i = 0; i < e.num_pieces; ++i)
        pieces[e.piece_sq[i]] = e.piece[i];
    if (e.kings[WHITE] != e.kings[BLACK]) {
        pieces[e.kings[WHITE]] = W_KING;
        pieces[e.kings[BLACK]] = B_KING;
    }

    for (int r = 7; r >= 0; --r) {
        int empty_counter = 0;
        for (int f = 0; f < 8; ++f) {
            int s = 8 * r + f;
            Piece p = pieces[s];
            if (type_of(p) == NONE) {
                ++empty_counter;
                continue;
            }

            if (empty_counter) {
                *buffer++ = '0' + empty_counter;
                empty_counter = 0;
            }

            *buffer++ = piece_char[p];
        }

        if (empty_counter)
            *buffer++ = '0' + empty_counter;
        if (r)
            *buffer++ = '/';
    }

    *buffer++ = ' ';
    *buffer++ = e.stm == WHITE ? 'w' : 'b';
    *buffer++ = ' ';

    *buffer++ = 0;
    return buffer;
}

