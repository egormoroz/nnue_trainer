#include <iostream>
#include <fstream>
#include "stream.hpp"
#include <memory>
#include <chrono>
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
    unique_ptr<SparseBatch> sb(new SparseBatch);
    BatchStream s("games.bin");

    size_t i = 0;
    for (int epoch = 0; epoch < 100; ++epoch) {
        while (s.next_batch(*sb)) {
            cout << (i++) << "\n";
        }
        s.reset();
    }
}

