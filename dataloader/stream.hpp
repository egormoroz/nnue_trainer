#ifndef STREAM_HPP
#define STREAM_HPP

#include <iosfwd>
#include <vector>

#include "bitutil.hpp"
#include "batch.hpp"
#include "fen.hpp"

struct OStream {
    OStream(std::ostream &os);

    //pieces must be sorted by square (from A1 to H8)
    //popcnt(mask) == n_pieces
    //Dumps the buffer automatically
    void write_entry(int16_t score, const Board &b, 
            GameResult result);

    //dumps the bytes of the buffer to the ostream
    //NB: doesn't write the last byte if it's incomplete
    void flush();

    //dumps the whole buffer (which is always padded with zeros)
    void flush_padded();

    ~OStream();

private:
    std::ostream &os_;
    uint8_t buffer_[10240];
    BitWriter writer_;
};

class IStream {
public:
    IStream(std::istream &is);

    void fetch_data_lazily();
    
    void decode_entry(TrainingEntry &e);
    void reset();

private:
    Piece decode_piece();

    std::istream &is_;

    std::vector<uint8_t> buffer_;
    BitReader reader_;
};

#endif
