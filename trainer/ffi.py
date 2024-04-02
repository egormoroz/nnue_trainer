import ctypes
import os

import torch
import numpy as np

dllmod = None


def load_module(dll_path):
    global dllmod
    dllmod = ctypes.cdll.LoadLibrary(os.path.abspath(dll_path))
    setup_prototypes(dllmod)


def setup_prototypes(dll):
    dll.create_batch_stream.restype = ctypes.c_void_p
    dll.create_batch_stream.argtypes = [
        ctypes.c_char_p, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int
    ]

    dll.destroy_batch_stream.restype = None
    dll.destroy_batch_stream.argtypes = [ctypes.c_void_p]

    dll.next_batch.restype = SparseBatchPtr
    dll.next_batch.argtypes = [ctypes.c_void_p]


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

        wft_ics = torch.from_numpy(wft_ics).to(device, non_blocking=True)
        bft_ics = torch.from_numpy(bft_ics).to(device, non_blocking=True)

        stm = torch.from_numpy(stm).pin_memory().to(device, non_blocking=True)
        score = torch.from_numpy(score).pin_memory().to(device, non_blocking=True)
        result = torch.from_numpy(result).pin_memory().to(device, non_blocking=True)

        return wft_ics, bft_ics, stm, score, result

    def __repr__(self) -> str:
        return f'SparseBatch(size={self.size}, max_active_fts={self.max_active_fts})'


SparseBatchPtr = ctypes.POINTER(SparseBatch)


class BatchStream:
    def __init__(self, bin_fpath: str, n_prefetch: int, n_workers: int,
                 batch_size: int):
        assert dllmod
        self.stream = dllmod.create_batch_stream(
                bin_fpath.encode('utf-8'), n_prefetch, n_workers, batch_size)

    def next_batch(self):
        assert dllmod
        return dllmod.next_batch(self.stream).contents


    def __del__(self):
        assert dllmod
        dllmod.destroy_batch_stream(self.stream)


