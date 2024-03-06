#ifndef BATCHSTREAM_HPP
#define BATCHSTREAM_HPP

#include <queue>
#include <mutex>
#include <thread>
#include <atomic>

#include "batch.hpp"

template<typename T>
class Queue {
public:
    Queue(size_t max_size)
        : max_size_(max_size), stop_(false)
    {}

    void push(T &&t) {
        std::unique_lock<std::mutex> lck(m_);
        while (q_.size() >= max_size_ && !stop_)
            non_full_.wait(lck);
        if (stop_) return;

        q_.push(std::move(t));
        lck.unlock();

        non_empty_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lck(m_);
        while (q_.empty())
            non_empty_.wait(lck);

        T val = std::move(q_.front());
        q_.pop();
        lck.unlock();

        non_full_.notify_one();
        return val;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lck(m_);
            stop_ = true;
        }
        non_full_.notify_all();
        non_empty_.notify_all();
    }

private:
    size_t max_size_;
    bool stop_;

    std::condition_variable non_empty_;
    std::condition_variable non_full_;
    std::mutex m_;
    std::queue<T> q_;
};


class BatchStream {
public:
    BatchStream(const char* fpath, int n_prefetch, 
            int batch_size, bool add_virtual);

    // (!) the previous batch is destroyed.
    // EOF is denoted by an empty batch
    SparseBatch* next_batch();

    ~BatchStream();

private:
    void worker_routine(const char *fpath, int batch_size, bool add_virtual);

    std::atomic_bool exit_;
    std::thread worker_;
    char worker_file_[256];

    Queue<SparseBatch> q_;
    SparseBatch cur_batch_;
};


#endif
