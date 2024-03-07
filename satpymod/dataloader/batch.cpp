#include "batch.hpp"


SparseBatch::SparseBatch(int batch_size, int max_fts) 
    : size(batch_size), max_active_fts(max_fts)
{
    if (!size)
        return;

    stm = new float[size];
    score = new float[size];
    result = new float[size];
    wfts = new int[size * max_active_fts];
    bfts = new int[size * max_active_fts];
}

void SparseBatch::fill(const TrainingEntry *entries, int n_entries) {
    size = n_entries;

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

void SparseBatch::free() {
    delete[] stm;
    delete[] score;
    delete[] result;
    delete[] wfts;
    delete[] bfts;
    size = 0;
}

