#pragma once

#include <cstdint>
#include "common.hpp"
#include "fen.hpp"

struct XorShift {
    XorShift()
        : XorShift(0xDEADBEEF, 0xB16B00B5) {}

    explicit XorShift(uint64_t x1, uint64_t x2)
        : state_{x1, x2} {}

    uint64_t operator()() {
        uint64_t t = state_[0];
        uint64_t const s = state_[1];
        state_[0] = s;
        t ^= t << 23;
        t ^= t >> 18;
        t ^= s ^ (s >> 5);
        state_[1] = t;
        return t + s;
    }

private:
    uint64_t state_[2];
};

struct Zobrist {
    uint64_t psq[2][6][64];
    uint64_t stm[2];
    uint64_t outcome[3];

    Zobrist() {
        XorShift rng;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 6; ++j)
                for (size_t k = 0; k < 64; ++k)
                    psq[i][j][k] = rng();
        stm[0] = rng();
        stm[1] = rng();

        outcome[0] = rng();
        outcome[1] = rng();
        outcome[2] = rng();
    }
};

uint64_t comp_hash(const Board &b, 
        GameResult outcome, int16_t score);

struct TrainingEntry;
uint64_t comp_hash(const TrainingEntry &e);


