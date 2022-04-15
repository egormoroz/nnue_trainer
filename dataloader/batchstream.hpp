#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <vector>

#include "bitutil.hpp"
#include "batch.hpp"


class BatchStream {
public:
    BatchStream(const char *file);

    bool eof();
    void reset();

    bool next_batch(SparseBatch &batch);

    ~BatchStream();

private:
    void worker_routine();

    void read_batch(SparseBatch &batch);

    void decode_entry(TrainingEntry &e);
    Piece decode_piece();

    void fetch_data();


    std::ifstream fin_;
    std::vector<uint8_t> buffer_;
    BitReader reader_;
    std::vector<TrainingEntry> entry_buf_;


    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool quit_{false}, reset_{false};

    std::vector<SparseBatch> batches_;
    bool eof_{false};
};


