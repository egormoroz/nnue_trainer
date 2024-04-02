#ifndef PYMOD_HPP
#define PYMOD_HPP

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT extern "C" __attribute__((visibility("default")))
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif


class BatchStream;
struct SparseBatch;

EXPORT BatchStream* create_batch_stream(const char *bin_fpath, int n_prefetch, 
        int n_workers, int batch_size);

EXPORT void destroy_batch_stream(BatchStream *bs);

EXPORT SparseBatch* next_batch(BatchStream *bs);

#endif
