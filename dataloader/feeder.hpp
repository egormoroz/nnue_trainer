#ifndef FEEDER_HPP
#define FEEDER_HPP

#include <fstream>
#include "stream.hpp"
#include "batch.hpp"

#define DLL_EXPORT __declspec(dllexport)


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

extern "C" DLL_EXPORT BinWriter* binwriter_new(const char* file);
extern "C" DLL_EXPORT BinReader* binreader_new(const char* file);

extern "C" DLL_EXPORT void delete_binwriter(BinWriter*);
extern "C" DLL_EXPORT void delete_binreader(BinReader*);

extern "C" DLL_EXPORT int write_entry(BinWriter *writer,
    const char *fen, int score);
extern "C" DLL_EXPORT int next_batch(BinReader *reader);

extern "C" DLL_EXPORT SparseBatch* get_batch(BinReader *reader);

#endif
