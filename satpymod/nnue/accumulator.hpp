#pragma once

#include <cstdint>
#include "../config.hpp"
#include "nnarch.hpp"

struct Accumulator {
#ifndef NONNUE
    int16_t v[2][nnspecs::HALFKP];
    int32_t psqt[2];
#endif
    bool computed[2];
};

