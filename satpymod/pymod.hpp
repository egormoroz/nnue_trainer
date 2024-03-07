#ifndef PYMOD_HPP
#define PYMOD_HPP

#include "pack.hpp"
#include <fstream>

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    //  GCC
    #define extern "C" EXPORT __attribute__((visibility("default")))
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif


struct PackReader {
    std::ifstream fin;
    ChainReader reader;
    char fen_buf[256]{};

    bool is_ok;
};

constexpr int RET_INVALID = 0xdeb14;

//EXPORT void dll_init();

EXPORT PackReader* pr_create(const char *file_path);
EXPORT void pr_reset(PackReader *pr);
EXPORT void pr_destroy(PackReader *pr);

EXPORT int pr_next(PackReader *pr);
EXPORT const char* pr_cur_fen(PackReader *pr);

EXPORT int pr_cur_score(PackReader *pr);
EXPORT int pr_cur_result(PackReader *pr);

EXPORT uint64_t pr_cur_hash(PackReader *pr);

EXPORT int pr_cur_eval(PackReader *pr);
EXPORT int pr_cur_nneval(PackReader *pr);

EXPORT int validate_pack(const char *fname, uint64_t hash);


class BatchStream;
struct SparseBatch;

EXPORT BatchStream* create_batch_stream(
        const char *bin_fpath, const char *index_fpath,
        int n_prefetch, int n_workers, int batch_size, int add_virtual);

EXPORT void destroy_batch_stream(BatchStream *bs);

EXPORT SparseBatch* next_batch(BatchStream *bs);

#endif
