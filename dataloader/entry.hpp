#ifndef ENTRY_HPP
#define ENTRY_HPP

#include <cstdint>
#include "common.hpp"

struct TrainingEntry {
    static constexpr int MAX_COMPRESSED_SIZE = 32;

    //in centipawns 
    int16_t score;

    uint8_t piece_sq[30];
    Piece piece[30];
    uint8_t kings[2];

    uint8_t num_pieces;
    Color stm;

    GameResult result;
};

#endif
