#ifndef FEEDER_HPP
#define FEEDER_HPP

#include <fstream>
#include "stream.hpp"
#include "batch.hpp"

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
};

struct BinReader {
    BinReader(const char *file);

    std::ifstream fin;
    IStream is;

    SparseBatch batch;
};

struct Features {
    int n_wfts{}, n_bfts{};
    float stm{};
    int wft_indices[MAX_ACTIVE_FEATURES],
        bft_indices[MAX_ACTIVE_FEATURES];
};

extern "C" EXPORT BinWriter* binwriter_new(const char* file);
extern "C" EXPORT BinReader* binreader_new(const char* file);

extern "C" EXPORT void delete_binwriter(BinWriter*);
extern "C" EXPORT void delete_binreader(BinReader*);

extern "C" EXPORT int write_entry(BinWriter *writer,
    const char *fen, int score, int result);
extern "C" EXPORT int next_batch(BinReader *reader);

extern "C" EXPORT SparseBatch* get_batch(BinReader *reader);

extern "C" EXPORT int reset_binreader(BinReader* reader);

extern "C" EXPORT Features* get_features(const char *fen);
extern "C" EXPORT void destroy_features(Features *fts);

#endif
