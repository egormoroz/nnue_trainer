#include "batchstream.hpp"
#include <cstring>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cmath>

#include <random>

#include "../pack.hpp"

constexpr inline float score_result_prob(Color stm, int score, int result) {
    const int remap_result[2][3] = { 
        {  1, -1, 0 },
        { -1,  1, 0 },
    };

    // make the score relative
    result = remap_result[stm][result];

    const float theta = 150.f;

    const float w_prob = 1 / (1 + std::exp(-score / theta));
    const float l_prob = 1 / (1 + std::exp(score / theta));
    const float d_prob = 1 - w_prob - l_prob;
    
    if (result > 0)
        return w_prob;
    else if (result < 0)
        return l_prob;
    else
        return d_prob;
}


BatchStream::BatchStream(const char* bin_fpath, int n_prefetch, 
        int n_workers, int bs, bool add_virtual, bool wait_on_end) 
    : batch_size_(bs), add_virtual_(add_virtual), n_workers_(n_workers),
      exit_(false), n_chunks_processed_(0), wait_on_end_(wait_on_end),
      chunk_queue_(n_workers), batch_queue_(n_prefetch)
{
    te_buffer_.reserve(batch_size_ * n_workers);
    strcpy(bin_fpath_, bin_fpath);

    cur_batch_ = allocate_batch();

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

void BatchStream::stop() {
    exit_ = true;
    chunk_queue_.stop();
    batch_queue_.stop();

    chunk_done_.notify_all();
}

void BatchStream::file_reader_routine() {
    std::ifstream fin_pack(bin_fpath_, std::ios::binary);
    if (!fin_pack) {
        fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: "
                "could not open file %s\n", bin_fpath_);
        return;
    }

    fin_pack.seekg(0, std::ios::end);
    size_t file_size = fin_pack.tellg();
    fin_pack.clear();
    fin_pack.seekg(0);

    if (file_size < ChunkHead::SIZE) {
        fprintf(stderr, "[ERROR] BatchStream::file_reader_routine: no chunks found\n");
        stop();
        return;
    }

    std::vector<size_t> chunk_offs;
    chunk_offs.reserve((file_size + PACK_CHUNK_SIZE - 1) / PACK_CHUNK_SIZE);
    for (size_t i = 0; i < file_size; i += PACK_CHUNK_SIZE)
        chunk_offs.push_back(i);

    std::default_random_engine rng(uint32_t(std::time(0)));
    std::uniform_int_distribution<size_t> dist(0, chunk_offs.size() - 1);

    while (!exit_) {
        Chunk ch = allocate_chunk();

        size_t off = chunk_offs[dist(rng)];
        fin_pack.clear();
        fin_pack.seekg(off);

        fin_pack.read(ch.data, PACK_CHUNK_SIZE);
        ch.size = fin_pack.gcount();
        chunk_queue_.push(ch);
    }
}

void BatchStream::worker_routine() {
    ChainReader2 cr;
    TrainingEntry te;
    std::vector<TrainingEntry> thread_te_buf;
    thread_te_buf.reserve(2 * batch_size_);

    std::default_random_engine rng(uint32_t(std::time(0)));

    while (!exit_) {
        Chunk ch;
        if (!chunk_queue_.pop(ch))
            break;

        ChunkHead head;
        head.from_bytes((const uint8_t*)ch.data);
        assert(head.SIZE + head.body_size <= PACK_CHUNK_SIZE);

        uint64_t hash = 0;
        size_t buf_size = ch.size;
        const uint8_t *ptr = (const uint8_t*)ch.data + head.SIZE;

        for (uint32_t i = 0; i < head.n_chains; ++i) {
            PackResult pr = cr.start_new_chain(ptr, buf_size);
            if (!is_ok(pr)) {
                fprintf(stderr, "[ERROR] BatchStream::worker_routine: encountered invalid "
                        "chain start, skipping the whole chunk...\n");
                goto skip_chunk;
            }

            for (uint16_t j = 0; j < cr.n_moves; ++j, pr = cr.next()) {
                if (!is_ok(pr)) {
                    fprintf(stderr, "[ERROR] BatchStream::worker_routine: encountered invalid "
                            "move in chain, skipping the whole chunk...\n");
                    goto skip_chunk;
                }
                hash ^= cr.board.key();


                const float score_prob = score_result_prob(cr.board.side_to_move(), cr.score, cr.result);
                std::bernoulli_distribution dist(1 - score_prob);
                bool skip_pos = cr.board.checkers() || !cr.board.is_quiet(cr.move) || dist(rng);

                if (skip_pos)
                    continue;

                te.score = cr.score;
                te.stm = cr.board.side_to_move();
                te.result = cr.result;

                te.n_wfts = halfkp::get_active_features(cr.board, WHITE, te.wfts);
                te.n_bfts = halfkp::get_active_features(cr.board, BLACK, te.bfts);

                if (add_virtual_) {
                    te.n_wfts += halfkp::get_virtual_active_features(
                            cr.board, WHITE, te.wfts + te.n_wfts);
                    te.n_bfts += halfkp::get_virtual_active_features(
                            cr.board, BLACK, te.bfts + te.n_bfts);
                }

                thread_te_buf.push_back(te);
            }

            assert(pr == PackResult::END_OF_CHAIN);
            hash ^= cr.board.do_move(cr.move).key();

            buf_size -= cr.tellg();
            ptr += cr.tellg();

            while (thread_te_buf.size() >= (size_t)batch_size_) {
                SparseBatch sb = allocate_batch();
                sb.fill(thread_te_buf.data() + thread_te_buf.size() - batch_size_, batch_size_);
                batch_queue_.push(sb);
                thread_te_buf.resize(thread_te_buf.size() - batch_size_);
            }
        }

        if (hash != head.hash)
            fprintf(stderr, "[WARNINIG] BatchStream::worker_routine: chunk hash mismatch.\n");
        if ((const char*)ptr - ch.data != head.body_size + head.SIZE)
            fprintf(stderr, "[WARNINIG] BatchStream::worker_routine: chunk size mismatch.\n");

        /* if (!exit_ && wait_on_end_) { */
        /*     collect_leftovers(thread_te_buf.data(), thread_te_buf.size()); */
        /*     thread_te_buf.clear(); */

        /*     ++n_chunks_processed_; */
        /*     chunk_done_.notify_all(); */
        /* } */

skip_chunk:
        free_chunk(ch);
    }
}

void BatchStream::collect_leftovers(const TrainingEntry *entries, 
        size_t n_entries, bool flush_nonfull) 
{
    std::unique_lock<std::mutex> lck(te_buffer_mtx_);
    if (n_entries)
        te_buffer_.insert(te_buffer_.end(), entries, entries + n_entries);

    if ((flush_nonfull && !te_buffer_.empty()) || te_buffer_.size() >= size_t(batch_size_)) {
        int n = std::min(batch_size_, int(te_buffer_.size()));
        SparseBatch sb = allocate_batch();
        sb.fill(te_buffer_.data() + te_buffer_.size() - n, n);
        te_buffer_.resize(te_buffer_.size() - n);
        lck.unlock();

        batch_queue_.push(sb);
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
    ch.data = new char[PACK_CHUNK_SIZE + CHUNK_PADDING];
    ch.size = 0;
    return ch;
}

void BatchStream::free_chunk(Chunk ch) {
    std::lock_guard<std::mutex> lck(free_chunk_mtx_);
    free_chunks_.push_back(ch);
}

BatchStream::~BatchStream() {
    stop();

    fread_thread_.join();
    for (auto &th: worker_threads_)
        th.join();

    for (auto &i: free_batches_)
        i.free();
    for (auto &i: free_chunks_)
        i.free();

    auto free_pred = [](auto &x) { x.free(); };
    chunk_queue_.apply(free_pred);
    batch_queue_.apply(free_pred);
}

