#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <vector>

#include "bitutil.hpp"
#include "batch.hpp"
#include "stream.hpp"


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

    std::ifstream fin_;
    IStream is_;
    std::vector<TrainingEntry> entry_buf_;

    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool quit_{false}, reset_{false};

    std::vector<SparseBatch> batches_;
    bool eof_{false};
};


