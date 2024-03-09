#include "batch.hpp"
#include <algorithm>


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

        // Bottlenecked by this thingy (i.e. RAM throughput)
        std::copy(e.wfts, e.wfts + e.n_wfts, wft_slice);
        std::fill(wft_slice + e.n_wfts, wft_slice + max_active_fts, -1);
        std::copy(e.bfts, e.bfts + e.n_bfts, bft_slice);
        std::fill(bft_slice + e.n_bfts, bft_slice + max_active_fts, -1);
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

