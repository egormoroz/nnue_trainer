#include "pymod.hpp"
#include "core/eval.hpp"
#include "zobrist.hpp"
#include "movgen/attack.hpp"
#include "dataloader/batchstream.hpp"

#include "pack.hpp"

#include <fstream>
#include <vector>
#include <cassert>

void dll_init() {
    init_zobrist();
    init_attack_tables();
    init_ps_tables();
#ifndef NONNUE
    if (!nnue::load_parameters("saturn.bin")) {
        printf("failed to initialize nnue, aborting\n");
    }
#endif
}

struct DLLINIT {
    DLLINIT() {
        dll_init();
    }
};
static DLLINIT __dllinit;

BatchStream* create_batch_stream(const char *bin_fpath, int n_prefetch, 
        int n_workers, int batch_size)
{
    assert(n_prefetch > 0 && batch_size > 0 && bin_fpath && n_workers > 0);
    return new BatchStream(bin_fpath, n_prefetch, n_workers, batch_size);
}

void destroy_batch_stream(BatchStream *bs) {
    assert(bs);
    delete bs;
}
                                                     
SparseBatch* next_batch(BatchStream *bs) {
    assert(bs);
    return bs->next_batch();
}

template<int block_size>
void compress(BitWriter &bw, const int16_t *begin, const int16_t *end) {
    constexpr uint16_t block_mask = (1 << block_size) - 1;

    for (; begin != end; ++begin) {
        uint16_t ux = abs(*begin);
        ux = ux << 1 | (*begin >= 0 ? 0 : 1);

        while (true) {
            bw.write(ux & block_mask, block_size);
            ux >>= block_size;

            if (ux) bw.write(1, 1);
            else break;
        }
        bw.write(0, 1);
    }
}

template<int block_size>
void decompress(BitReader &br, int16_t *params, int n_params) {
    for (int i = 0; i < n_params; ++i) {
        int off = 0;
        uint16_t x = 0;
        do {
            x |= br.read<uint16_t>(block_size) << off;
            off += block_size;

        } while (br.read<uint8_t>(1));

        int16_t sign = (x & 1) == 0 ? 1 : -1;

        params[i] = int16_t(x >> 1) * sign;
    }
}

constexpr int block_size = 4;

int compress_net(const char* path_in, const char* path_out) {
    std::ifstream fin(path_in, std::ios::binary);
    if (!fin.is_open())  {
        fprintf(stderr, "Failed to open %s\n", path_in);
        return 1;
    }

    fin.seekg(0, std::ios::end);

    uint32_t n_params = fin.tellg();
    if (n_params % 2 != 0)  {
        fprintf(stderr, "Expected an input file with flat i16 array\n");
        return 1;
    }
    n_params /= 2;

    fin.clear();
    fin.seekg(0);

    std::vector<int16_t> orig(n_params);
    fin.read((char*)orig.data(), n_params * 2);

    std::vector<uint8_t> compressed(n_params * 2, 0);
    BitWriter bw { compressed.data(), 0 };

    compress<block_size>(bw, orig.data(), orig.data() + n_params);
    size_t n_compressed_bytes = (bw.cursor + 7) / 8;

    compressed.resize(n_compressed_bytes);
    std::ofstream fout(path_out, std::ios::binary);
    if (!fout.is_open()) {
        fprintf(stderr, "Failed to create %s\n", path_in);
        return 1;
    }
    fout.write((const char*)&n_params, sizeof(n_params));
    fout.write((const char*)compressed.data(), n_compressed_bytes);

    return 0;
}

int decompress_net(const char* path_in, const char* path_out) {
    std::ifstream fin(path_in, std::ios::binary);
    if (!fin.is_open())  {
        fprintf(stderr, "Failed to open %s\n", path_in);
        return 1;
    }

    uint32_t n_params;
    if (!fin.read((char*)&n_params, sizeof(n_params))) {
        fprintf(stderr, "Failed to read n_params (first %d bytes) %s\n", 
                int(sizeof(n_params)), path_in);
        return 1;
    }

    size_t off = fin.tellg();
    fin.seekg(0, std::ios::end);
    size_t n_compressed_bytes = fin.tellg();
    fin.clear();
    fin.seekg(off);

    std::vector<uint8_t> compressed(n_compressed_bytes);
    fin.read((char*)compressed.data(), n_compressed_bytes);

    std::vector<int16_t> restored(n_params);
    BitReader br { compressed.data(), 0 };

    decompress<block_size>(br, restored.data(), n_params);
    std::ofstream fout(path_out, std::ios::binary);
    if (!fout.is_open()) {
        fprintf(stderr, "Failed to create %s\n", path_in);
        return 1;
    }

    fout.write((const char*)restored.data(), n_params * 2);

    return 0;
}

