#include "batch.hpp"
#include <cstring>


SparseBatch::SparseBatch(const TrainingEntry *entries, 
        int n_entries, bool with_virtual) 
{
    size = n_entries;

    if (!size)
        return;

    stm = new float[size];
    score = new float[size];
    result = new float[size];

    max_active_fts = with_virtual ? halfkp::MAX_TOTAL_FTS : halfkp::MAX_REAL_FTS;

    wfts = new int[size * max_active_fts];
    bfts = new int[size * max_active_fts];

    /* constexpr float res_to_val[3] = { 0.f, 1.f, 0.5f, }; */
    constexpr float res_to_val[2][3] = { 
        { 1.f, 0.f, 0.5f }, //white to move
        { 0.f, 1.f, 0.5f }, //black to move
    };

    for (int i = 0; i < n_entries; ++i) {
        const TrainingEntry &e = entries[i];
        stm[i] = e.stm;
        score[i] = e.score;
        result[i] = res_to_val[e.stm][e.result];

        int* wft_slice = wfts + i * max_active_fts;
        int* bft_slice = bfts + i * max_active_fts;
        int s;

        for (s = 0; s < (int)e.n_wfts; ++s)
            wft_slice[s] = e.wfts[s];
        for (; s < max_active_fts; ++s)
            wft_slice[s] = -1;

        for (s = 0; s < (int)e.n_bfts; ++s)
            bft_slice[s] = e.bfts[s];
        for (; s < max_active_fts; ++s)
            bft_slice[s] = -1;
    }
}

SparseBatch::SparseBatch(SparseBatch &&other) {
    memcpy(this, &other, sizeof(SparseBatch));
    other.size = 0;
}

SparseBatch& SparseBatch::operator=(SparseBatch &&other) {
    if (this->size) {
        delete[] stm;
        delete[] score;
        delete[] result;
        delete[] wfts;
        delete[] bfts;
    }

    memcpy(this, &other, sizeof(SparseBatch));
    other.size = 0;
    return *this;
}


SparseBatch::~SparseBatch() {
    if (size) {
        delete[] stm;
        delete[] score;
        delete[] result;
        delete[] wfts;
        delete[] bfts;
    }
}

