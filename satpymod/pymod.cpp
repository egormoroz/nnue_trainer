#include "pymod.hpp"
#include "core/eval.hpp"
#include "nnue/evaluate.hpp"
#include "pack.hpp"
#include "zobrist.hpp"
#include "movgen/attack.hpp"
#include "dataloader/batchstream.hpp"

#include <fstream>
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

PackReader* pr_create(const char *file_path) {
    std::ifstream fin(file_path, std::ios::binary);
    if (!fin)
        return nullptr;

    PackReader *pr = new PackReader;
    pr->fin = std::move(fin);
    pr->is_ok = is_ok(pr->reader.start_new_chain(pr->fin));
    return pr;
}

void pr_reset(PackReader *pr) {
    if (!pr)
        return;

    pr->fin.seekg(0);
    pr->fin.clear();
    pr->is_ok = is_ok(pr->reader.start_new_chain(pr->fin));
}

void pr_destroy(PackReader *pr) {
    delete pr;
}
                                                     
int pr_next(PackReader *pr) {
    if (!pr)
        return RET_INVALID;

    PackResult res = pr->reader.next(pr->fin);
    if (is_ok(res))
        pr->is_ok = is_ok(res);
    else if (res == PackResult::END_OF_CHAIN)
        pr->is_ok = is_ok(pr->reader.start_new_chain(pr->fin));
    return pr->is_ok ? 0 : 1;
}

const char* pr_cur_fen(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return nullptr;

    pr->reader.board.get_fen(pr->fen_buf);
    return pr->fen_buf;
}

int pr_cur_score(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return RET_INVALID;
    int score = pr->reader.score;
    return pr->reader.board.side_to_move() == WHITE ? score : -score;
}

int pr_cur_result(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return RET_INVALID;
    return pr->reader.result;
}

uint64_t pr_cur_hash(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return RET_INVALID;
    return pr->reader.board.key();
}
                                                     
int pr_cur_eval(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return RET_INVALID;
    return evaluate(pr->reader.board);
}

int pr_cur_nneval(PackReader *pr) {
    if (!pr || !pr->is_ok)
        return RET_INVALID;
#ifdef NONNUE
    return RET_INVALID;
#endif
    nnue::refresh_accumulator(pr->reader.board, pr->reader.si.acc, WHITE);
    nnue::refresh_accumulator(pr->reader.board, pr->reader.si.acc, BLACK);
    return nnue::evaluate(pr->reader.board);
}
                                                     
int validate_pack(const char *fname, uint64_t hash) {
    assert(fname);
    return 1 - validate_packed_games(fname, hash);
}

BatchStream* create_batch_stream(
        const char *bin_fpath, const char *index_fpath,
        int n_prefetch, int n_workers, int batch_size, int add_virtual)
{
    assert(n_prefetch > 0 && batch_size > 0 && bin_fpath && index_fpath && n_workers > 0);
    return new BatchStream(bin_fpath, index_fpath, 
            n_prefetch, n_workers, batch_size, add_virtual);
}
                                                     
void destroy_batch_stream(BatchStream *bs) {
    assert(bs);
    delete bs;
}
                                                     
SparseBatch* next_batch(BatchStream *bs) {
    assert(bs);
    return bs->next_batch();
}
