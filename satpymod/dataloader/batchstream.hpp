#ifndef BATCHSTREAM_HPP
#define BATCHSTREAM_HPP

#include <condition_variable>
#include <deque>
#include <vector>
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

    template<typename U>
    void push(U &&u) {
        std::unique_lock<std::mutex> lck(m_);
        while (q_.size() >= max_size_ && !stop_)
            non_full_.wait(lck);
        if (stop_) return;

        q_.push_back(std::forward<U>(u));
        lck.unlock();

        non_empty_.notify_one();
    }

    bool pop(T &t) {
        std::unique_lock<std::mutex> lck(m_);
        while (q_.empty() && !stop_)
            non_empty_.wait(lck);

        if (stop_)
            return false;

        t = std::move(q_.front());
        q_.pop_front();
        lck.unlock();

        non_full_.notify_one();
        return true;
    }

    template<typename F>
    void apply(F &&f) {
        std::lock_guard<std::mutex> lck(m_);
        for (auto &i: q_)
            f(i);
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
    std::deque<T> q_;
};

class BatchStream {
public:
    BatchStream(const char* bin_fpath, int n_prefetch, int n_workers, 
            int batch_size, bool add_virtual, bool wait_on_end);

    // (!) the previous batch is destroyed.
    // EOF is denoted by an empty batch
    SparseBatch* next_batch();
    void reset();

    void stop();

    ~BatchStream();

private:
    struct Chunk {
        // Always exactly max_chunk_size_ bytes allocated
        char *data = nullptr;
        size_t size = 0;

        void free() {
            if (data)
                delete[] data;
        }
    };


    void file_reader_routine();
    void worker_routine();

    void collect_leftovers(const TrainingEntry *entries, size_t n_entries,
            bool flush_nonfull = false);

    SparseBatch allocate_batch();
    void free_batch(SparseBatch b);

    Chunk allocate_chunk();
    void free_chunk(Chunk ch);

    const int batch_size_;
    const bool add_virtual_;

    const int n_workers_;

    char bin_fpath_[256];

    std::atomic_bool exit_;
    // This gets incremented ONLY after all data has been successfully pushed to the batch queue.
    // Thus we can gurantee than next_batch has seen all data exactly once 
    // if n_chunks_processed_ == index_.n_blocks
    std::atomic_uint64_t n_chunks_processed_;
    // TODO: better name needed. 
    // This flag basically means if we are in the "infinite" mode, where we loop through 
    // the whole dataset infinitely. If we need to do a single pass
    // through the whole dataset, then this flag is true.
    const bool wait_on_end_;

    std::mutex epoch_done_mtx_;
    std::condition_variable chunk_done_;

    std::mutex te_buffer_mtx_;
    std::vector<TrainingEntry> te_buffer_;

    Queue<Chunk> chunk_queue_;
    Queue<SparseBatch> batch_queue_;
    SparseBatch cur_batch_;

    std::mutex free_batch_mtx_;
    std::vector<SparseBatch> free_batches_;
    std::mutex free_chunk_mtx_;
    std::vector<Chunk> free_chunks_;

    std::thread fread_thread_;
    std::vector<std::thread> worker_threads_;
};


#endif
