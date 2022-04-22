#include "batch.hpp"
#include <algorithm>

//helpers
namespace {

template<bool mirror>
void add_features(const TrainingEntry &e, int sample_idx,
        int *features, int &features_nb, int ksq)
{
    for (int i = 0;i  < e.num_pieces; ++i) {
        Piece p = e.piece[i];

        int psq = e.piece_sq[i];
        if constexpr (mirror)
            psq = sq_mirror(psq);

        features[features_nb * 2] = sample_idx;
        features[features_nb * 2 + 1] = halfkp_idx(ksq, psq, p);
        ++features_nb;
    }
}

} //namespace

void SparseBatch::fill(const TrainingEntry *entries, int n) {
    size = std::min(n, MAX_SIZE);

    n_active_white_fts = 0;
    n_active_black_fts = 0;

    for (int i = 0; i < size; ++i) {
        const TrainingEntry &e = entries[i];

        stm[i] = e.stm == WHITE;
        score[i] = e.score;
        result[i] = +e.result == +e.stm;
        result[i] += float(e.result == DRAW) * 0.5f;

        add_features<false>(e, i, white_fts_indices, 
                n_active_white_fts, e.kings[WHITE]);
        add_features<true>(e, i, black_fts_indices, 
                n_active_black_fts, sq_mirror(e.kings[BLACK]));
    }
}

