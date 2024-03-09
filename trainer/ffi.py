import ctypes
import os

import torch
import numpy as np

dllmod = None

_MOD_RET_INVALID = 0xdeb14


def load_module(dll_path):
    global dllmod
    dllmod = ctypes.cdll.LoadLibrary(os.path.abspath(dll_path))
    setup_prototypes(dllmod)


def setup_prototypes(dll):
    dll.pr_create.restype = ctypes.c_void_p
    dll.pr_create.argtypes = [ctypes.c_char_p]

    dll.pr_destroy.restype = None
    dll.pr_destroy.argtypes = [ctypes.c_void_p]

    dll.pr_reset.restype = None
    dll.pr_reset.argtypes = [ctypes.c_void_p]

    dll.pr_next.restype = ctypes.c_int
    dll.pr_next.argtypes = [ctypes.c_void_p]

    dll.pr_cur_fen.restype = ctypes.c_char_p
    dll.pr_cur_fen.argtypes = [ctypes.c_void_p]

    dll.pr_cur_score.restype = ctypes.c_int
    dll.pr_cur_score.argtypes = [ctypes.c_void_p]

    dll.pr_cur_result.restype = ctypes.c_int
    dll.pr_cur_result.argtypes = [ctypes.c_void_p]

    dll.pr_cur_hash.restype = ctypes.c_uint64
    dll.pr_cur_hash.argtypes = [ctypes.c_void_p]

    dll.pr_cur_eval.restype = ctypes.c_int
    dll.pr_cur_eval.argtypes = [ctypes.c_void_p]

    dll.pr_cur_nneval.restype = ctypes.c_int
    dll.pr_cur_nneval.argtypes = [ctypes.c_void_p]

    dll.validate_pack.restype = ctypes.c_int
    dll.validate_pack.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong]

    dll.create_batch_stream.restype = ctypes.c_void_p
    dll.create_batch_stream.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    dll.destroy_batch_stream.restype = None
    dll.destroy_batch_stream.argtypes = [ctypes.c_void_p]

    dll.next_batch.restype = SparseBatchPtr
    dll.next_batch.argtypes = [ctypes.c_void_p]


class PackReader:
    def __init__(self, bin_path: str):
        assert dllmod
        self.pr = dllmod.pr_create(bin_path.encode('utf-8'))
        assert self.pr, 'file not found probably'

    def reset(self):
        assert dllmod and self.pr
        dllmod.pr_reset(self.pr)

    def next(self) -> bool:
        assert dllmod and self.pr
        return True if dllmod.pr_next(self.pr) == 0 else False

    def cur_fen(self) -> str:
        assert dllmod and self.pr
        return dllmod.pr_cur_fen(self.pr).decode('utf-8')

    def cur_score(self) -> int:
        assert dllmod and self.pr
        return dllmod.pr_cur_score(self.pr)

    def cur_result(self) -> int:
        assert dllmod and self.pr
        return dllmod.pr_cur_result(self.pr)

    def cur_hash(self) -> int:
        assert dllmod and self.pr
        return dllmod.pr_cur_hash(self.pr)

    def cur_eval(self) -> int:
        assert dllmod and self.pr
        return dllmod.pr_cur_eval(self.pr)

    def cur_nneval(self) -> int | None:
        assert dllmod and self.pr
        x = dllmod.pr_cur_nneval(self.pr)
        return x if x != _MOD_RET_INVALID else None

    def __del__(self):
        assert dllmod and self.pr
        dllmod.pr_destroy(self.pr)


def validate_pack(bin_path: str, hash_or_path: int | str):
    assert dllmod
    if isinstance(hash_or_path, str):
        with open(hash_or_path) as f:
            hash_or_path = int(f.read())
    ret = dllmod.validate_pack(bin_path.encode('utf-8'), hash_or_path)
    return True if ret == 0 else False


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('max_active_fts', ctypes.c_int),

        ('stm', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('result', ctypes.POINTER(ctypes.c_float)),

        ('wfts', ctypes.POINTER(ctypes.c_int)),
        ('bfts', ctypes.POINTER(ctypes.c_int)),
    ]

    def to_numpy(self):
        wft_ics = np.ctypeslib.as_array(self.wfts, shape=(self.size, self.max_active_fts))
        bft_ics = np.ctypeslib.as_array(self.bfts, shape=(self.size, self.max_active_fts))
        stm = np.ctypeslib.as_array(self.stm, shape=(self.size, 1))
        score = np.ctypeslib.as_array(self.score, shape=(self.size, 1))
        result = np.ctypeslib.as_array(self.result, shape=(self.size, 1))
        return wft_ics, bft_ics, stm, score, result


    def to_torch(self, device='cpu'):
        wft_ics, bft_ics, stm, score, result = self.to_numpy()

        wft_ics = torch.from_numpy(wft_ics).to(device)
        bft_ics = torch.from_numpy(bft_ics).to(device)

        wft_vals = torch.ones_like(wft_ics, dtype=torch.float32)
        bft_vals = torch.ones_like(bft_ics, dtype=torch.float32)

        stm = torch.from_numpy(stm).to(device)
        score = torch.from_numpy(score).to(device)
        result = torch.from_numpy(result).to(device)

        return wft_ics, wft_vals, bft_ics, bft_vals, stm, score, result

    def __repr__(self) -> str:
        return f'SparseBatch(size={self.size}, max_active_fts={self.max_active_fts})'


SparseBatchPtr = ctypes.POINTER(SparseBatch)


class BatchStream:
    def __init__(self, bin_fpath: str, index_fpath: str,
                 n_prefetch: int, n_workers: int,
                 batch_size: int, add_virtual: int):
        assert dllmod
        self.stream = dllmod.create_batch_stream(
                bin_fpath.encode('utf-8'), index_fpath.encode('utf-8'), 
                n_prefetch, n_workers, batch_size, add_virtual)

    def next_batch(self):
        assert dllmod
        batch = dllmod.next_batch(self.stream).contents
        return batch if batch.size > 0 else None


    def __del__(self):
        assert dllmod
        dllmod.destroy_batch_stream(self.stream)


