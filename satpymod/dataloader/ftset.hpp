#ifndef FTSET_HPP
#define FTSET_HPP

#include "../board/board.hpp"

namespace mini {

constexpr int N_FEATURES = 12 * 64;
constexpr int MAX_TOTAL_FTS = 32;

constexpr uint16_t index(Color pov, Square sq, Piece p) {
    const uint16_t p_idx = 2 * (type_of(p) - 1) + uint16_t(color_of(p) != pov);
    const uint16_t o_sq = sq ^ (pov == WHITE ? 0 : 56);
    return p_idx * 64 + o_sq;
}



inline int get_active_features(const Board &b, Color side, uint16_t *fts) {
    int n_fts = 0;
    Bitboard mask = b.pieces();
    while (mask) {
        Square psq = pop_lsb(mask);
        fts[n_fts++] = index(side, psq, b.piece_on(psq));
    }

    return n_fts;
}


} // mini


#endif
