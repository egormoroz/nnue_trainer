#include "batchstream.hpp"
#include <chrono>
#include <cstring>

namespace huffman {

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

}

constexpr int N = SparseBatch::MAX_SIZE * TrainingEntry::MAX_COMPRESSED_SIZE;
constexpr size_t PREFETCH_NB = 8;

BatchStream::BatchStream(const char *file)
    : fin_(file, std::ios::binary),
      buffer_(N * 2, 0), reader_{ buffer_.data(), 0 },
      worker_([this]() { worker_routine(); })
{
}

bool BatchStream::eof() { 
    std::lock_guard<std::mutex> lock(mtx_);
    return batches_.empty() &&  eof_; 
}

void BatchStream::reset() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        reset_ = true;
        eof_ = false;
    }
    cv_.notify_one();
}

bool BatchStream::next_batch(SparseBatch &batch) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (batches_.empty()) {
        if (eof_)
            return false;

        //worker thread can't keep up so we have to wait
        lock.unlock();
        cv_.notify_one();

        while (true) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(10));

            lock.lock();

            if (!batches_.empty()) break;
            if (eof_) return false;

            lock.unlock();
        }
        
        //mutex is already locked
    }

    batch = batches_.back();
    batches_.pop_back();

    lock.unlock();
    cv_.notify_one();

    return true;
}

void BatchStream::worker_routine() {
    std::vector<SparseBatch> batch_buf;

    fin_.read((char*)buffer_.data(), buffer_.size());
    entry_buf_.reserve(SparseBatch::MAX_SIZE);

    while (true) {
        size_t n = batch_buf.size();
        batch_buf.resize(PREFETCH_NB);

        for (size_t i = n; i < PREFETCH_NB; ++i) {
            read_batch(batch_buf[i]);

            int size = batch_buf[i].size;
            n += size != 0;
            if (!size)
                break;
        }
        batch_buf.resize(n);

        std::unique_lock<std::mutex> lock(mtx_);
        if (!n) eof_ = true;
        cv_.wait(lock, [&]() {
            return quit_ || reset_ || batches_.size() < n;
        });

        if (quit_)
            break;

        if (reset_) {
            reset_ = false;
            fin_.clear();
            fin_.seekg(0, fin_.beg);
            
            fin_.read((char*)buffer_.data(), buffer_.size());
            reader_.cursor = 0;

            batches_.clear();
            batch_buf.clear();
            continue;
        }

        std::swap(batch_buf, batches_);
    }
}

BatchStream::~BatchStream() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        quit_ = true;
    }
    cv_.notify_one();
    worker_.join();
}

void BatchStream::read_batch(SparseBatch &batch) {
    batch.size = 0;

    if (reader_.cursor > N * 8)
        fetch_data();

    TrainingEntry e;
    entry_buf_.clear();
    for (int i = 0; i < SparseBatch::MAX_SIZE; ++i) {
        decode_entry(e);
        if (!e.num_pieces) {
            break;
        }

        entry_buf_.push_back(e);
    }

    batch.fill(entry_buf_.data(), entry_buf_.size());
}

void BatchStream::decode_entry(TrainingEntry &e) {
    memset(&e, 0, sizeof(e));

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

Piece BatchStream::decode_piece() {
    Color c = Color(reader_.read_bit());
    uint8_t code = 0;

    while(true) {
        code |= reader_.read_bit();
        if (huffman::d_table[code] != NONE)
            return make_piece(c, huffman::d_table[code]);
        code <<= 1;
    }
}

void BatchStream::fetch_data() {
    assert(reader_.cursor > N * 8);

    size_t cur_byte = reader_.cursor / 8;
    size_t bytes_left = 2 * N - cur_byte;

    memcpy(buffer_.data(), buffer_.data() + cur_byte, bytes_left);
    reader_.cursor %= 8;

    fin_.read((char*)&buffer_[bytes_left], 2 * N - bytes_left);
    size_t n = fin_.gcount();

    if (n < 2 * N - bytes_left)
        memset(&buffer_[bytes_left + n], 0, 2 * N - bytes_left - n);
}

