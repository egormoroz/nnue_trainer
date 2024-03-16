#include "pymod.hpp"
#include "core/eval.hpp"
#include "zobrist.hpp"
#include "movgen/attack.hpp"
#include "dataloader/batchstream.hpp"

#include <cassert>

void dll_init() {
    init_zobrist();
    init_attack_tables();
    init_ps_tables();
#ifndef NONNUE
    if (!nnue::load_parameters("saturn.bin")) {
        printf("failed to initialize nnue, aborting\n");
    }
#endif
}

struct DLLINIT {
    DLLINIT() {
        dll_init();
    }
};
static DLLINIT __dllinit;

BatchStream* create_batch_stream(
        const char *bin_fpath, int n_prefetch, int n_workers, 
        int batch_size, int add_virtual, int wait_on_end)
{
    assert(n_prefetch > 0 && batch_size > 0 && bin_fpath && n_workers > 0);
    return new BatchStream(bin_fpath, n_prefetch, n_workers, batch_size, add_virtual, wait_on_end);
}

void destroy_batch_stream(BatchStream *bs) {
    assert(bs);
    delete bs;
}
                                                     
SparseBatch* next_batch(BatchStream *bs) {
    assert(bs);
    return bs->next_batch();
}
