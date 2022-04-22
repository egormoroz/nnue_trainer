#include "batchstream.hpp"
#include <chrono>
#include <cstring>


constexpr size_t PREFETCH_NB = 8;

BatchStream::BatchStream(const char *file)
    : fin_(file, std::ios::binary), is_(fin_),
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
            is_.reset();

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
    is_.fetch_data_lazily();

    TrainingEntry e;
    entry_buf_.clear();
    for (int i = 0; i < SparseBatch::MAX_SIZE; ++i) {
        is_.decode_entry(e);
        if (!e.num_pieces)
            break;

        entry_buf_.push_back(e);
    }

    batch.fill(entry_buf_.data(), entry_buf_.size());
}

