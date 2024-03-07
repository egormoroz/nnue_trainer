#ifndef BATCH_HPP
#define BATCH_HPP

#include "halfkp.hpp"

struct TrainingEntry {
    static_assert(halfkp::N_FT + halfkp::N_VIRT_FT <= 0xFFFF);

    uint16_t wfts[halfkp::MAX_TOTAL_FTS];
    uint16_t bfts[halfkp::MAX_TOTAL_FTS];

    uint8_t n_wfts;
    uint8_t n_bfts;

    int16_t score;
    Color stm;
    uint8_t result;
};

// Always has the capacity of batch_size bytes
struct SparseBatch {
    SparseBatch() = default;
    SparseBatch(int batch_size, int max_active_fts);

    void fill(const TrainingEntry *entries, int n_entries);
    void free();

    int size = 0;
    int max_active_fts = 0;

    float *stm = nullptr;
    float *score = nullptr;
    float *result = nullptr;

    int *wfts = nullptr;
    int *bfts = nullptr;
};

#endif
