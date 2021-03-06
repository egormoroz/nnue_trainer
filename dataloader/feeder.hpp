#ifndef FEEDER_HPP
#define FEEDER_HPP

#include <fstream>
#include "stream.hpp"
#include "batch.hpp"
#include "batchstream.hpp"

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif


struct BinWriter {
    BinWriter(const char *file);

    std::ofstream fout;
    OStream os;
    uint64_t hash = 0;
};

struct Features {
    int n_wfts{}, n_bfts{};
    float stm{};
    int wft_indices[MAX_ACTIVE_FEATURES],
        bft_indices[MAX_ACTIVE_FEATURES];
};

extern "C" EXPORT BinWriter* binwriter_new(const char* file);

extern "C" EXPORT int write_entry(BinWriter *writer,
    const char *fen, int score, int result);

extern "C" EXPORT uint64_t binwriter_get_hash(BinWriter *writer);

extern "C" EXPORT void delete_binwriter(BinWriter*);


extern "C" EXPORT BatchStream* batchstream_new(const char* file);
extern "C" EXPORT void delete_batchstream(BatchStream*);
extern "C" EXPORT int next_batch(
    BatchStream *stream, SparseBatch *batch);
extern "C" EXPORT void reset_batchstream(BatchStream* stream);

extern "C" EXPORT SparseBatch* new_batch();
extern "C" EXPORT void destroy_batch(SparseBatch *batch);

extern "C" EXPORT Features* get_features(const char *fen);
extern "C" EXPORT void destroy_features(Features *fts);


extern "C" EXPORT uint64_t bin_comp_hash(const char *file);

#endif
