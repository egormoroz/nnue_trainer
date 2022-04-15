#include "feeder.hpp"
#include "fen.hpp"
#include <algorithm>

BinWriter::BinWriter(const char *file)
    : fout(file, std::ios::binary), os(fout)
{}

BinReader::BinReader(const char *file)
    : fin(file, std::ios::binary), is(fin)
{}


BinWriter* binwriter_new(const char* file) {
    return new BinWriter(file);
}

BatchStream* batchstream_new(const char* file) {
    return new BatchStream(file);
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

    writer->os.write_entry(static_cast<int16_t>(score),
        b.stm, b.mask, b.pieces, b.n_pieces, GameResult(result));

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

static void add_indices(const Board &b, int *indices, 
        int &n, int ksq) 
{
    uint64_t mask = b.mask;
    for (int i = 0; i < b.n_pieces; ++i) {
        Piece p = b.pieces[i];
        int psq = pop_lsb(mask);

        if (type_of(p) == KING) continue;

        indices[n++] = halfkp_idx2(ksq, psq, p);
    }

    std::sort(indices, indices + n);
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

    add_indices(b, fts->wft_indices, fts->n_wfts, ksq[WHITE]);
    add_indices(b, fts->bft_indices, fts->n_bfts, ksq[BLACK]);

    return fts;
}

void destroy_features(Features *fts) {
    delete fts;
}

