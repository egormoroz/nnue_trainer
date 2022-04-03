#ifndef STREAM_HPP
#define STREAM_HPP

#include <iosfwd>
#include <vector>
#include "bitutil.hpp"
#include "batch.hpp"

struct OStream {
    OStream(std::ostream &os);

    //pieces must be sorted by square (from A1 to H8)
    //popcnt(mask) == n_pieces
    //Dumps the buffer automatically
    void write_entry(int16_t score, Color stm, uint64_t mask, 
            const Piece *pieces, int n_pieces);

    //dumps the bytes of the buffer to the ostream
    //NB: doesn't write the last byte if it's incomplete
    void flush();

    //dumps the whole buffer (which is always padded with zeros)
    void flush_padded();

    ~OStream();

private:
    std::ostream &os_;
    uint8_t buffer_[1024];
    BitWriter writer_;
};


struct IStream {
    IStream(std::istream &is);

    bool eof() const;

    void read_batch(SparseBatch &batch);

    void decode_entry(TrainingEntry &e);
    Piece decode_piece();

    void fetch_data();

private:
    std::istream &is_;
    std::vector<uint8_t> buffer_;
    BitReader reader_;

    std::vector<TrainingEntry> entries_;

    bool eof_{false};
};

#endif
