import ctypes as ct
import atexit

BATCH_SIZE = 32768
MAX_ACTIVE_FEATURES = 32

class SparseBatch(ct.Structure):
    _fields_ = [
        ('size', ct.c_int),
        ('n_wfts', ct.c_int),
        ('n_bfts', ct.c_int),

        ('stm', ct.c_float * BATCH_SIZE),
        ('score', ct.c_float * BATCH_SIZE),
        ('result', ct.c_float * BATCH_SIZE),

        ('wft_indices', 
            ct.c_int * (BATCH_SIZE * MAX_ACTIVE_FEATURES * 2)),
        ('bft_indices', 
            ct.c_int * (BATCH_SIZE * MAX_ACTIVE_FEATURES * 2)),
    ]


SparseBatchPtr = ct.POINTER(SparseBatch)


class Features(ct.Structure):
    _fields_ = [
        ('n_wfts', ct.c_int),
        ('n_bfts', ct.c_int),

        ('stm', ct.c_float),
        ('wft_indices', ct.c_int * MAX_ACTIVE_FEATURES),
        ('bft_indices', ct.c_int * MAX_ACTIVE_FEATURES),
    ]


def load_dll(dll_path: str):
    dll = ct.cdll.LoadLibrary(dll_path)

    dll.binwriter_new.argtypes = (ct.c_char_p,)
    dll.binwriter_new.restype = ct.c_void_p

    dll.delete_binwriter.argtypes = (ct.c_void_p,)
    dll.delete_binwriter.restype = None

    dll.write_entry.argtypes = (ct.c_void_p, ct.c_char_p,
            ct.c_int, ct.c_int)
    dll.write_entry.restype = ct.c_int


    dll.batchstream_new.argtypes = (ct.c_char_p,)
    dll.batchstream_new.restype = ct.c_void_p

    dll.delete_batchstream.argtypes = (ct.c_void_p,)
    dll.delete_batchstream.restype = None

    dll.next_batch.argtypes = (ct.c_void_p, SparseBatchPtr)
    dll.next_batch.restype = ct.c_int

    dll.reset_batchstream.argtypes = (ct.c_void_p,)
    dll.reset_batchstream.restype = None


    dll.new_batch.argtypes = None
    dll.new_batch.restype = SparseBatchPtr

    dll.destroy_batch.argtypes = (SparseBatchPtr,)
    dll.destroy_batch.restype = None


    dll.get_features.argtypes = (ct.c_char_p,)
    dll.get_features.restype = ct.POINTER(Features)

    dll.destroy_features.argtypes = (ct.POINTER(Features),)
    dll.destroy_features.restype = None


    return dll


class BinWriter:
    def __init__(self, dll, file_path: str):
        self.dll = dll
        path = file_path.encode('utf-8')
        self.writer = self.dll.binwriter_new(path)
        atexit.register(self.cleanup)

    def cleanup(self):
        self.dll.delete_binwriter(self.writer)
        
    def write_entry(self, fen: str, score: int, result: int) -> int:
        fen = fen.encode('utf-8')
        return self.dll.write_entry(self.writer, fen, score, result)


class BatchStream:
    def __init__(self, dll, file_path: str):
        file_path = file_path.encode('utf-8')

        self.dll = dll
        self.stream = dll.batchstream_new(file_path)
        self.batch = dll.new_batch()

        atexit.register(self.cleanup)

    def get_batch(self) -> SparseBatch:
        return self.batch.contents

    def next_batch(self) -> int:
        return self.dll.next_batch(self.stream, self.batch)

    def reset(self):
        self.dll.reset_batchstream(self.stream)        

    def cleanup(self):
        self.dll.delete_batchstream(self.stream)
        self.dll.destroy_batch(self.batch)


