from ffi import *
import torch

class SparseBatchIter:
    def __init__(self, reader: BinReader):
        self.reader = reader
        self.reader.reset()

    def __iter__(self):
        return self

    def __next__(self) -> SparseBatch:
        if self.reader.next_batch() == 0:
            raise StopIteration
        return self.reader.get_batch()


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dll, filepath):
        self.reader = BinReader(dll, filepath)

    def __iter__(self):
        return SparseBatchIter(self.reader)


