#include "batchstream.hpp"
#include <vector>
#include <fstream>
#include <cstdio>

#include "../pack.hpp"
#include "membuf.hpp"

/* TODO:
 * 1. Compute max buffer sizes
 * 2. Preallcoate buffers
 * 3. Something else but I forgor
 * */


BatchStream::BatchStream(const char* bin_fpath, const char* index_fpath,
        int n_prefetch, int n_workers, int bs, bool add_virtual) 
    : batch_size_(bs), add_virtual_(add_virtual), n_workers_(n_workers),
      exit_(false), n_chunks_processed_(0),
      chunk_queue_(n_workers), batch_queue_(n_prefetch)
{
    strcpy_s(bin_fpath_, bin_fpath);
    strcpy_s(index_fpath_, index_fpath);

    std::ifstream fin_index(index_fpath_, std::ios::binary);
    if (!fin_index) {
        fprintf(stderr, "[WARNING] BatchStream::file_reader_routine: index file %s not found, "
                "creating one from scratch. This might take a while...\n", index_fpath_);
        if (!create_index(bin_fpath_, index_)) {
            fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: "
                    "index file could not be created...\n");
            return;
        }

        std::ofstream fout(index_fpath_, std::ios::binary);
        index_.write_to_stream(fout);

    } else if (!index_.load_from_stream(fin_index)) {
        fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: index file %s is invalid, "
                "PackIndex::load_from_stream failed...\n", index_fpath_);
        return;
    }

    cur_batch_ = allocate_batch();

    max_chunk_size_ = 0;
    for (size_t i = 0; i < index_.n_blocks; ++i) {
        auto &blk = index_.blocks[i];
        max_chunk_size_ = std::max(max_chunk_size_, int(blk.off_end - blk.off_begin));
    }

    fread_thread_ = std::thread([this] { file_reader_routine(); });
    worker_threads_.resize(n_workers_);
    for (int i = 0; i < n_workers_; ++i)
        worker_threads_[i] = std::thread([this] { worker_routine(); });
}

SparseBatch* BatchStream::next_batch() {
    free_batch(cur_batch_);
    bool b = batch_queue_.pop(cur_batch_);
    assert(b); // no reason for this to be false
    return b ? &cur_batch_ : nullptr;
}

void BatchStream::file_reader_routine() {
    std::ifstream fin_pack(bin_fpath_, std::ios::binary);
    if (!fin_pack) {
        fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: "
                "could not open file %s\n", bin_fpath_);
        return;
    }

    size_t blk_idx = 0;
    while (!exit_) {
        auto &blk = index_.blocks[blk_idx++];
        std::streamsize blk_size = blk.off_end - blk.off_begin;
        assert(max_chunk_size_ >= blk_size);

        // this should be a no-op, because the blocks are sequential
        Chunk ch = allocate_chunk();
        ch.size = blk_size;
        fin_pack.seekg(blk.off_begin);
        fin_pack.read(ch.data, blk_size);
        if (fin_pack.gcount() < blk_size) {
            fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: "
                    "couldn\'t read a whole block\n");
            return;
        }

        chunk_queue_.push(ch);

        if (blk_idx >= index_.n_blocks) {
            fin_pack.clear();
            blk_idx = 0;

            // let's make sure all data has been processed before starting new round
            {
                std::unique_lock<std::mutex> lck(epoch_done_mtx_);
                while (n_chunks_processed_ < index_.n_blocks)
                    chunk_done_.wait(lck);
            }
            n_chunks_processed_ = 0;
        }
    }
}

void BatchStream::worker_routine() {
    ChainReader r;
    std::vector<TrainingEntry> thread_te_buf;
    thread_te_buf.reserve(batch_size_);

    while (!exit_) {
        Chunk ch;
        if (!chunk_queue_.pop(ch))
            break;
            
        membuf buf(ch.data, ch.data + ch.size);
        std::istream is(&buf);

        PackResult pr;
        TrainingEntry te;
        while (!exit_ && is_ok(pr = r.start_new_chain(is))) {
            do {
                bool skip_pos = r.board.checkers() || !r.board.is_quiet(r.move);
                if (skip_pos)
                    continue;

                te.score = r.score;
                te.stm = r.board.side_to_move();
                te.result = r.result;

                te.n_wfts = halfkp::get_active_features(r.board, WHITE, te.wfts);
                te.n_bfts = halfkp::get_active_features(r.board, BLACK, te.bfts);

                if (add_virtual_) {
                    te.n_wfts += halfkp::get_virtual_active_features(
                            r.board, WHITE, te.wfts + te.n_wfts);
                    te.n_bfts += halfkp::get_virtual_active_features(
                            r.board, BLACK, te.bfts + te.n_bfts);
                }

                thread_te_buf.push_back(te);
            } while (is_ok(pr = r.next(is)));

            if (pr != PackResult::END_OF_CHAIN) {
                fprintf(stderr, "BatchStream::worker_routine: "
                        "encountered incorrect chain[1], aborting...\n");
                free_chunk(ch);
                return;
            }

            if ((int)thread_te_buf.size() >= batch_size_) {
                on_new_entries(thread_te_buf.data(), thread_te_buf.size());
                thread_te_buf.clear();
            }
        }
        free_chunk(ch);
        if (exit_) break;

        if (pr != PackResult::END_OF_FILE) {
            fprintf(stderr, "[WARNING] BatchStream::worker_routine: "
                    "final chain was incorrect...\n");
        } else if (!thread_te_buf.empty()) {
            on_new_entries(thread_te_buf.data(), thread_te_buf.size());
            thread_te_buf.clear();
        }

        ++n_chunks_processed_;
        chunk_done_.notify_all();
    }
}

void BatchStream::on_new_entries(const TrainingEntry *entries, int n_entries) {
    std::lock_guard<std::mutex> lck(te_buffer_mtx_);
    while (n_entries > 0) {
        int n = batch_size_ - int(te_buffer_.size());
        assert(n >= 0);

        n = std::min(n, n_entries);
        te_buffer_.insert(te_buffer_.end(), entries, entries + n);
        entries += n;
        n_entries -= n;

        if (int(te_buffer_.size()) == batch_size_) {
            SparseBatch sb = allocate_batch();
            sb.fill(te_buffer_.data(), batch_size_);
            te_buffer_.clear();
            batch_queue_.push(sb);
        }
    }
}

SparseBatch BatchStream::allocate_batch() {
    std::unique_lock<std::mutex> lck(free_batch_mtx_);
    if (!free_batches_.empty()) {
        SparseBatch b = free_batches_.back();
        free_batches_.pop_back();
        return b;
    }
    lck.unlock();

    const int max_active_fts = add_virtual_ ? halfkp::MAX_TOTAL_FTS : halfkp::MAX_REAL_FTS;
    return SparseBatch(batch_size_, max_active_fts);
}

void BatchStream::free_batch(SparseBatch b) {
    std::lock_guard<std::mutex> lck(free_batch_mtx_);
    free_batches_.push_back(b);
}

BatchStream::Chunk BatchStream::allocate_chunk() {
    std::unique_lock<std::mutex> lck(free_chunk_mtx_);
    if (!free_chunks_.empty()) {
        Chunk ch = free_chunks_.back();
        free_chunks_.pop_back();
        return ch;
    }
    lck.unlock();

    Chunk ch;
    ch.data = new char[max_chunk_size_];
    ch.size = 0;
    return ch;
}

void BatchStream::free_chunk(Chunk ch) {
    std::lock_guard<std::mutex> lck(free_chunk_mtx_);
    free_chunks_.push_back(ch);
}

BatchStream::~BatchStream() {
    exit_ = true;
    chunk_queue_.stop();
    batch_queue_.stop();

    fread_thread_.join();
    for (auto &th: worker_threads_)
        th.join();

    for (auto &i: free_batches_)
        i.free();
    for (auto &i: free_chunks_)
        i.free();
}

