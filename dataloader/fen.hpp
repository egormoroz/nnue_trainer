#ifndef FEN_HPP
#define FEN_HPP

#include "common.hpp"
#include <string_view>
#include "entry.hpp"

struct Board {
    uint64_t mask;
    Piece pieces[32];
    int n_pieces;
    Color stm;
};

bool parse_fen(std::string_view fen, Board &b);
char* get_fen(const TrainingEntry &e, char *buffer);

#endif
