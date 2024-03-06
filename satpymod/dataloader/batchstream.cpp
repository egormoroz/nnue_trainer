#include "batchstream.hpp"
#include <vector>
#include <fstream>
#include <cstdio>

#include "../pack.hpp"
#include "../board/board.hpp"
#include "../movgen/generate.hpp"


BatchStream::BatchStream(const char* fpath, int n_prefetch, 
        int bs, bool add_virtual) 
    : exit_(false), 
      q_(n_prefetch)
{
    strcpy_s(worker_file_, fpath);
    worker_ = std::thread([=]{ worker_routine(worker_file_, bs, add_virtual); });
}

SparseBatch* BatchStream::next_batch() {
    cur_batch_ = q_.pop();
    return &cur_batch_;
}


void BatchStream::worker_routine(const char *fpath, 
        int batch_size, bool add_virtual) 
{
    std::vector<TrainingEntry> buffer;
    buffer.reserve(batch_size + 256);

    std::ifstream fin(fpath, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "[ERROR] BatchStream::worker_routine: "
                "could not open file %s\n", fpath);
        return;
    }

    ChainReader r;
    TrainingEntry te;
    bool loaded_any = false;

    do {
        // fetch next batch...
        bool eof = false;
        while ((int)buffer.size() < batch_size) {
            if (!is_ok(r.start_new_chain(fin))) {
                fin.clear();
                fin.seekg(0);
                eof = true;
                break;
            }
            loaded_any = true;

            do {
                bool skip_pos = r.board.checkers() || !r.board.is_quiet(r.move);
                if (skip_pos)
                    continue;

                te.score = r.score;
                te.stm = r.board.side_to_move();
                te.result = r.result;

                te.n_wfts = halfkp::get_active_features(r.board, WHITE, te.wfts);
                te.n_bfts = halfkp::get_active_features(r.board, BLACK, te.bfts);

                if (add_virtual) {
                    te.n_wfts += halfkp::get_virtual_active_features(
                            r.board, WHITE, te.wfts + te.n_wfts);
                    te.n_bfts += halfkp::get_virtual_active_features(
                            r.board, BLACK, te.bfts + te.n_bfts);
                }

                buffer.push_back(te);
            } while (is_ok(r.next(fin)));
        }

        int n = std::min((int)buffer.size(), batch_size);
        q_.push(SparseBatch(buffer.data(), n, add_virtual));
        buffer.erase(buffer.begin(), buffer.begin() + n);

        if (eof && n) {
            // push empty batch as a sign of EOF
            q_.push(SparseBatch(nullptr, 0, false));
        }

    } while (!exit_ && loaded_any);

    if (!loaded_any) {
        fprintf(stderr, "[ERROR] BatchStream::worker_routine: "
                "could not load an entry %s\n", fpath);
    }
}


BatchStream::~BatchStream() {
    exit_ = true;
    q_.stop();
    if (worker_.joinable())
        worker_.join();
}
