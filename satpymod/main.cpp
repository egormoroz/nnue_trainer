#include <cstdio>
#include "config.hpp"
#include "pymod.hpp"
#include "dataloader/batchstream.hpp"

#include <chrono>


int main() {

#ifdef NONNUE
    printf("NNUE is disabled, using regular eval\n");
#else
#include "nnue/evaluate.hpp"
    if (!nnue::load_parameters("saturn.bin")) {
        printf("failed to initialize nnue, aborting\n");
        return 1;
    }
#endif

    using clk_t = std::chrono::steady_clock;
    constexpr char path[] = "d6v6_200mil_2.bin";

    BatchStream stream(path, 1, 8, 1024*16, false, false);

    auto start = clk_t::now();
    int batch_per_sec = 0;
    for (int i = 0; ; ++i) {
        SparseBatch *sb = stream.next_batch();

        if (i % 100 == 0) {
            printf("[%d] %d batch/s %d \n", i, batch_per_sec, sb->size);
            int delta = std::chrono::duration_cast<std::chrono::seconds>(
                    clk_t::now() - start).count();
            batch_per_sec = (i + 1) / std::max(1, delta);
        }
    }

    printf("done!\n");

    return 0;
}

