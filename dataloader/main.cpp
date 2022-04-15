#include <iostream>
#include <fstream>
#include "stream.hpp"
#include <memory>
#include <chrono>
#include <vector>
#include <numeric>
#include "batchstream.hpp"

using namespace std;

using Clock = chrono::steady_clock;

template<typename F>
int64_t time_it(F &&f) {
    auto start = Clock::now();
    f();
    return chrono::duration_cast<chrono::milliseconds>(
        Clock::now() - start
    ).count();
}

int main() {
}

