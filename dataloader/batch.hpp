#ifndef LIB_HPP
#define LIB_HPP

#include "common.hpp"
#include "entry.hpp"

struct SparseBatch {
    static constexpr int MAX_SIZE = 32768;

    void fill(const TrainingEntry *entries, int n_entries);

    int size{};
    int n_active_white_fts{},
        n_active_black_fts{};

    float stm[MAX_SIZE];
    float score[MAX_SIZE];
    float result[MAX_SIZE];

    int white_fts_indices[MAX_SIZE * MAX_ACTIVE_FEATURES * 2];
    int black_fts_indices[MAX_SIZE * MAX_ACTIVE_FEATURES * 2];
};


#endif
