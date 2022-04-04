#include "stream.hpp"
#include <istream>
#include <ostream>
#include <cstring>

namespace huffman {

struct Encoding {
    uint8_t bits;
    uint8_t n_bits;
};

constexpr Encoding e_table[] = {
    { 0b0,    1 }, //pawn
    { 0b111,  3 }, //knight
    { 0b001,  3 }, //bishop
    { 0b0011, 4 }, //rook
    { 0b1011, 4 }, //queen
    { 0b101,  3 }, //king
};

constexpr PieceType d_table[] = {
    PAWN, //0
    NONE, NONE, NONE,
    BISHOP, //4
    KING, //5
    NONE,
    KNIGHT, //7
    NONE, NONE, NONE, NONE,
    ROOK, //12
    QUEEN, //13
    NONE,
    NONE
};

} //namespace huffman



OStream::OStream(std::ostream &os)
    : os_(os), writer_ { &buffer_[0], 0 } 
{
    memset(buffer_, 0, sizeof(buffer_));
}

void OStream::write_entry(int16_t score, Color stm, uint64_t mask, 
        const Piece *pieces, int n_pieces, GameResult result) 
{
    if (writer_.cursor > (sizeof(buffer_) - 32) * 8)
        flush();

    writer_.write(score);
    writer_.write_bit(result);
    writer_.write_bit(result >> 1);
    writer_.write_bit(stm);
    writer_.write(mask);

    for (int i = 0; i < n_pieces; ++i) {
        Piece p = pieces[i];
        auto &encoding = huffman::e_table[type_of(p)];

        writer_.write_bit(color_of(p));
        writer_.write(encoding.bits, encoding.n_bits);
    }
}

void OStream::flush() {
    size_t byte_idx = writer_.cursor / 8,
        bit_idx = writer_.cursor % 8;
    os_.write((const char*)buffer_, byte_idx);

    uint8_t bits = buffer_[byte_idx];
    memset(buffer_, 0, sizeof(buffer_));

    writer_.cursor = bit_idx;
    buffer_[0] = bits;
}

void OStream::flush_padded() {
    os_.write((const char*)buffer_, sizeof(buffer_));
}

OStream::~OStream() {
    if (writer_.cursor)
        flush_padded();
}

constexpr int N = SparseBatch::MAX_SIZE * TrainingEntry::MAX_COMPRESSED_SIZE;

IStream::IStream(std::istream &is) 
    : is_(is), buffer_(N * 2, 0),
      reader_{ buffer_.data(), 0 }
{
    is_.read((char*)buffer_.data(), buffer_.size());
    entries_.reserve(SparseBatch::MAX_SIZE);
}

bool IStream::eof() const { return eof_; }

void IStream::read_batch(SparseBatch &batch) {
    batch.size = 0;

    if (reader_.cursor > N * 8)
        fetch_data();

    entries_.clear();
    TrainingEntry e;
    for (int i = 0; i < SparseBatch::MAX_SIZE; ++i) {
        decode_entry(e);
        if (!e.num_pieces) {
            eof_ = true;
            break;
        }

        entries_.push_back(e);
    }
    batch.fill(entries_.data(), entries_.size());
}

void IStream::decode_entry(TrainingEntry &e) {
    e.score = static_cast<int16_t>(reader_.read16());

    e.result = static_cast<GameResult>(
        reader_.read_bit() | reader_.read_bit() << 1);

    e.stm = Color(reader_.read_bit());
    e.num_pieces = 0;
    e.kings[WHITE] = e.kings[BLACK] = 0;

    uint64_t mask = reader_.read64();
    while (mask) {
        uint8_t sq = pop_lsb(mask);
        Piece p = decode_piece();

        if (type_of(p) == KING) {
            e.kings[color_of(p)] = sq;
            continue;
        }

        e.piece_sq[e.num_pieces] = sq;
        e.piece[e.num_pieces] = p;
        ++e.num_pieces;
    }
}

Piece IStream::decode_piece() {
    Color c = Color(reader_.read_bit());
    uint8_t code = 0;
    do {
        code |= reader_.read_bit();

        if (huffman::d_table[code] != NONE)
            return make_piece(c, huffman::d_table[code]);
        code <<= 1;
    } while (code < 16);

    assert(false);
    return W_NONE;
}

void IStream::fetch_data() {
    assert(reader_.cursor > N * 8);

    size_t cur_byte = reader_.cursor / 8;
    size_t bytes_left = 2 * N - cur_byte;

    memcpy(buffer_.data(), buffer_.data() + cur_byte, bytes_left);
    reader_.cursor %= 8;

    is_.read((char*)&buffer_[bytes_left], 2 * N - bytes_left);
    size_t n = is_.gcount();

    if (n < 2 * N - bytes_left)
        memset(&buffer_[bytes_left + n], 0, 2 * N - bytes_left - n);
}

