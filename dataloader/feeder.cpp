#include "feeder.hpp"
#include "fen.hpp"
#include <algorithm>
#include "myhash.hpp"
#include <fstream>

BinWriter::BinWriter(const char *file)
    : fout(file, std::ios::binary), os(fout)
{}

BinWriter* binwriter_new(const char* file) {
    return new BinWriter(file);
}

BatchStream* batchstream_new(const char* file) {
    return new BatchStream(file);
}

uint64_t binwriter_get_hash(BinWriter *writer) {
    return writer->hash;
}

void delete_binwriter(BinWriter* w) {
    delete w;
}

void delete_batchstream(BatchStream *stream) { delete stream; }

int write_entry(BinWriter *writer, const char *fen, 
        int score, int result) 
{
    if (!writer->fout.is_open())
        return 0;

    Board b;
    if (!parse_fen(fen, b))
        return 0;

    GameResult outcome = GameResult(result);
    int16_t score16 = static_cast<int16_t>(score);

    writer->os.write_entry(score16, b, outcome);
    writer->hash ^= comp_hash(b, outcome, score16);

    return 1;
}

int next_batch(BatchStream *stream, SparseBatch *batch) {
    return stream->next_batch(*batch);
}

void reset_batchstream(BatchStream* stream) {
    stream->reset();
}

SparseBatch* new_batch() { return new SparseBatch; }
void destroy_batch(SparseBatch *batch) { delete batch; }

namespace {

template<bool mirror>
void add_indices(const Board &b, int *indices, 
        int &n, int ksq) 
{
    uint64_t mask = b.mask;
    for (int i = 0; i < b.n_pieces; ++i) {
        Piece p = b.pieces[i];
        int psq = pop_lsb(mask);

        if (type_of(p) == KING) continue;

        if constexpr (mirror)
            psq = sq_mirror(psq);
        indices[n++] = halfkp_idx(ksq, psq, p);
    }

    std::sort(indices, indices + n);
}

}

Features* get_features(const char *fen) {
    Features *fts = new Features();

    Board b;
    if (!parse_fen(fen, b) || !b.n_pieces)
        return fts;

    int ksq[2]{};
    uint64_t mask = b.mask;
    for (int i = 0; i < b.n_pieces; ++i) {
        int s = pop_lsb(mask);
        Piece p = b.pieces[i];
        if (type_of(p) == KING)
            ksq[color_of(p)] = s;
    }

    if (!ksq[0] || !ksq[1])
        return fts;

    fts->stm = b.stm == WHITE;

    add_indices<false>(b, fts->wft_indices, 
            fts->n_wfts, ksq[WHITE]);
    add_indices<true>(b, fts->bft_indices, 
            fts->n_bfts, sq_mirror(ksq[BLACK]));

    return fts;
}

void destroy_features(Features *fts) {
    delete fts;
}


uint64_t bin_comp_hash(const char *file) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin.is_open())
        return 0;
    IStream stream(fin);

    uint64_t hash = 0;
    TrainingEntry e;
    int i = 0;
    while(true) {
        if (i % SparseBatch::MAX_SIZE == 0)
            stream.fetch_data_lazily();

        stream.decode_entry(e);
        if (!e.num_pieces)
            break;

        hash ^= comp_hash(e);
        ++i;
    }

    return hash;
}

