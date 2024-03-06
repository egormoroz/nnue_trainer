#include <cstdio>
#include "config.hpp"
#include "pymod.hpp"
#include "dataloader/batchstream.hpp"

#include <fstream>

using namespace std;

bool compare_seqs(const PosSeq &a, const PosSeq &b) {
    if (a.n_moves != b.n_moves)
        return false;

    for (int i = 0; i < a.n_moves; ++i)
        if (a.seq[i].move_idx != b.seq[i].move_idx || a.seq[i].score != b.seq[i].score)
            return false;
    return true;
}

int main() {
    /* init_zobrist(); */
    /* init_attack_tables(); */
    /* init_ps_tables(); */

#ifdef NONNUE
    printf("NNUE is disabled, using regular eval\n");
#else
#include "nnue/evaluate.hpp"
    if (!nnue::load_parameters("saturn.bin")) {
        printf("failed to initialize nnue, aborting\n");
        return 1;
    }
#endif

    /* repack_games("2games.bin", "2games_repack.bin"); */
    bool result = validate_packed_games("d7_repack.bin", "d7.hash");
    if (result) {
        printf("valid!!\n");
    } else {
        printf("invalid...\n");
    }

    return 0;
}

