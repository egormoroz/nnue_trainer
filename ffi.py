import ctypes as ct
import atexit

BATCH_SIZE = 256
MAX_ACTIVE_FEATURES = 32

class SparseBatchStruct(ct.Structure):
    _fields_ = [
        ('size', ct.c_int),
        ('n_active_white_fts', ct.c_int),
        ('n_active_black_fts', ct.c_int),

        ('stm', ct.c_float * BATCH_SIZE),
        ('score', ct.c_float * BATCH_SIZE),

        ('white_fts_indices', 
            ct.c_int * BATCH_SIZE * MAX_ACTIVE_FEATURES * 2),
        ('black_fts_indices', 
            ct.c_int * BATCH_SIZE * MAX_ACTIVE_FEATURES * 2),
    ]


def load_dll(dll_path: str):
    dll = ct.cdll.LoadLibrary(dll_path)

    dll.binwriter_new.argtypes = (ct.c_char_p,)
    dll.binwriter_new.restype = ct.c_void_p

    dll.binreader_new.argtypes = (ct.c_char_p,)
    dll.binreader_new.restype = ct.c_void_p

    dll.delete_binwriter.argtypes = (ct.c_void_p,)
    dll.delete_binwriter.restype = None

    dll.delete_binreader.argtypes = (ct.c_void_p,)
    dll.delete_binreader.restype = None

    dll.write_entry.argtypes = (ct.c_void_p, ct.c_char_p,
            ct.c_int)
    dll.write_entry.restype = ct.c_int

    dll.next_batch.argtypes = (ct.c_void_p,)
    dll.next_batch.restype = ct.c_int

    dll.get_batch.argtypes = (ct.c_void_p,)
    dll.get_batch.restype = ct.POINTER(SparseBatchStruct)

    return dll


class BinWriter:
    def __init__(self, dll, file_path: str):
        self.dll = dll
        path = file_path.encode('utf-8')
        self.writer = self.dll.binwriter_new(path)
        atexit.register(self.cleanup)

    def cleanup(self):
        self.dll.delete_binwriter(self.writer)
        
    def write_entry(self, fen: str, score: int) -> int:
        fen = fen.encode('utf-8')
        return self.dll.write_entry(self.writer, fen, score)


class BinReader:
    def __init__(self, dll, file_path: str):
        self.dll = dll
        path = file_path.encode('utf-8')
        self.reader = self.dll.binreader_new(path)
        atexit.register(self.cleanup)

    def get_batch(self):
        return self.dll.get_batch(self.reader)

    def next_batch(self):
        return self.dll.next_batch(self.reader)

    def cleanup(self):
        self.dll.delete_binreader(self.reader)

