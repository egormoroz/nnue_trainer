#include "myhash.hpp"
#include "entry.hpp"

static Zobrist zobrist;

uint64_t comp_hash(const Board &b, 
        GameResult outcome, int16_t score) 
{
    uint64_t hash = 0;

    uint64_t mask = b.mask;
    int i = 0;
    while (mask) {
        int s = pop_lsb(mask);
        Piece p = b.pieces[i++];

        hash ^= zobrist.psq[color_of(p)][type_of(p)][s];
    }

    hash ^= zobrist.stm[b.stm]
         ^ zobrist.outcome[outcome]
         ^ (uint64_t)score;

    return hash;
}

uint64_t comp_hash(const TrainingEntry &e) {
    uint64_t hash = 0;

    for (int i = 0; i < e.num_pieces; ++i) {
        Piece p = e.piece[i];
        int s = e.piece_sq[i];

        hash ^= zobrist.psq[color_of(p)][type_of(p)][s];
    }

    hash ^= zobrist.psq[WHITE][KING][e.kings[WHITE]];
    hash ^= zobrist.psq[BLACK][KING][e.kings[BLACK]];

    hash ^= zobrist.stm[e.stm]
         ^  zobrist.outcome[e.result]
         ^ (uint64_t)e.score; 

    return hash;
}


