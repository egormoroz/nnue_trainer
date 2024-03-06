#ifndef HALFLP_HPP
#define HALFLP_HPP

#include "../primitives/common.hpp"

class Board;

namespace halfkp {

constexpr int N_SQ = 64;
constexpr int N_PC = 10;
constexpr int N_FT = N_SQ * N_PC * N_SQ;

constexpr int N_VIRT_FT = N_SQ * N_PC;

constexpr int MAX_REAL_FTS = 30;
constexpr int MAX_VIRT_FTS = 30;
constexpr int MAX_TOTAL_FTS = MAX_REAL_FTS + MAX_VIRT_FTS;

constexpr inline Square orient(Color c, Square sq) {
    return Square(sq ^ (c == WHITE ? 0x0 : 0x3f));
}

constexpr int piece_to_index(Color pov, Piece p) {
    return 2 * (type_of(p) - 1) + int(color_of(p) != pov);
}

constexpr uint16_t halfkp_idx(Color pov, Square ksq, Square psq, Piece p) {
    return orient(pov, psq) + N_SQ*piece_to_index(pov, p) + N_SQ*N_PC*ksq;
}

int get_active_features(const Board &b, Color side, uint16_t *fts);
int get_virtual_active_features(const Board &b, Color side, uint16_t *fts);

}

#endif
