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

struct SparseBatch {
    // TODO: check perfomance and consider arena allocator
    // SF doesn't do this btw
    SparseBatch(const TrainingEntry *entries, int n_entries, bool with_virtual);

    SparseBatch() = default;
    SparseBatch(SparseBatch &&other);
    SparseBatch(const SparseBatch&) = delete;

    int size = 0;
    int max_active_fts;

    float *stm;
    float *score;
    float *result;

    int *wfts;
    int *bfts;

    SparseBatch& operator=(SparseBatch &&other);
    SparseBatch& operator=(const SparseBatch&) = delete;

    ~SparseBatch();
};

#endif
