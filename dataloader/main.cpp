#include <iostream>
#include <fstream>
#include "stream.hpp"
#include <memory>
#include <chrono>

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
    unique_ptr<SparseBatch> sb(new SparseBatch);
    ifstream fin("37540_games.bin", ios::binary);
    IStream is(fin);

    int batches = 84 * 12;
    int64_t millis = time_it([&]() {
        for (int i = 0; i < batches; ++i)
            is.read_batch(*sb);
    });

    int64_t n = SparseBatch::MAX_SIZE;
    n *= batches;
    n /= millis;
    cout << n << "k/s" << endl;
}

