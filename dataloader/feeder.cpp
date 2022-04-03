#include "feeder.hpp"
#include "fen.hpp"

BinWriter::BinWriter(const char *file)
    : fout(file, std::ios::binary), os(fout)
{}

BinReader::BinReader(const char *file)
    : fin(file, std::ios::binary), is(fin)
{}


BinWriter* binwriter_new(const char* file) {
    return new BinWriter(file);
}

BinReader* binreader_new(const char* file) {
    return new BinReader(file);
}

void delete_binwriter(BinWriter* w) {
    delete w;
}

void delete_binreader(BinReader* r) {
    delete r;
}

int write_entry(BinWriter *writer, const char *fen, int score) {
    if (!writer->fout.is_open())
        return 0;

    Board b;
    if (!parse_fen(fen, b))
        return 0;

    writer->os.write_entry(static_cast<int16_t>(score),
        b.stm, b.mask, b.pieces, b.n_pieces);

    return 1;
}

int next_batch(BinReader *reader) {
    if (!reader->fin.is_open())
        return 0;
    if (reader->is.eof())
        return 0;

    reader->is.read_batch(reader->batch);

    return 1;
}

SparseBatch* get_batch(BinReader *reader) {
    return &reader->batch;
}

