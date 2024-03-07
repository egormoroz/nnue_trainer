#include "halfkp.hpp"
#include "../board/board.hpp"

namespace halfkp {

int get_active_features(const Board &b, Color side, uint16_t *fts) {
    Square ksq = b.king_square(side);
    ksq = orient(side, ksq);

    int n_fts = 0;
    Bitboard mask = b.pieces() & ~b.pieces(KING);
    while (mask) {
        Square psq = pop_lsb(mask);
        fts[n_fts++] = halfkp_idx(side, ksq, psq, b.piece_on(psq));
    }

    return n_fts;
}


int get_virtual_active_features(const Board &b, Color side, uint16_t *fts) {
    int n_fts = 0;
    Bitboard mask = b.pieces() & ~b.pieces(KING);
    while (mask) {
        Square psq = pop_lsb(mask);
        int p_idx = piece_to_index(side, b.piece_on(psq));
        fts[n_fts++] = N_FT + orient(side, psq) + N_SQ * p_idx;
    }
    
    return n_fts;
}

}

