from ffi import *
import torch

class SparseBatchIter:
    def __init__(self, stream: BatchStream):
        self.stream = stream
        self.stream.reset()

    def __iter__(self):
        return self

    def __next__(self) -> SparseBatch:
        if self.stream.next_batch() == 0:
            raise StopIteration
        return self.stream.get_batch()


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dll, filepath):
        self.stream = BatchStream(dll, filepath)

    def __iter__(self):
        return SparseBatchIter(self.stream)


