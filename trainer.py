import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from ranger21 import Ranger21

from ffi import *
from dataset import *

NUM_FEATURES = 40960
SCALE = 500.0

def get_tensors(sb: SparseBatch):
    stm = np.ctypeslib.as_array(sb.stm)[:sb.size]
    stm = torch.from_numpy(stm.reshape((sb.size, 1)))

    score = np.ctypeslib.as_array(sb.score)[:sb.size]
    score = torch.from_numpy(score.reshape((sb.size, 1)))

    result = np.ctypeslib.as_array(sb.result)[:sb.size]
    result = torch.from_numpy(result.reshape((sb.size, 1)))

    wft_indices = np.ctypeslib.as_array(sb.wft_indices)
    wft_indices = wft_indices[:sb.n_wfts * 2].reshape((sb.n_wfts, 2)).T
    bft_indices = np.ctypeslib.as_array(sb.bft_indices)
    bft_indices = bft_indices[:sb.n_bfts * 2].reshape((sb.n_bfts, 2)).T

    wft_vals = torch.ones(sb.n_wfts)
    bft_vals = torch.ones(sb.n_bfts)

    w = torch._sparse_coo_tensor_unsafe(wft_indices, 
            wft_vals, (sb.size, NUM_FEATURES))
    b = torch._sparse_coo_tensor_unsafe(bft_indices, 
            bft_vals, (sb.size, NUM_FEATURES))

    w._coalesced_(True)
    b._coalesced_(True)

    return w, b, stm, score, result


def get_entry_tensors(fts):
    wft_indices = np.ctypeslib.as_array(fts.wft_indices)
    wft_indices = wft_indices[:fts.n_wfts].reshape((1, fts.n_wfts))
    wft_indices = np.concatenate((np.zeros_like(wft_indices), wft_indices))

    bft_indices = np.ctypeslib.as_array(fts.bft_indices)
    bft_indices = bft_indices[:fts.n_bfts].reshape((1, fts.n_bfts))
    bft_indices = np.concatenate((np.zeros_like(bft_indices), bft_indices))

    wft_vals = np.ones(fts.n_wfts, dtype=np.float32)
    bft_vals = np.ones(fts.n_bfts, dtype=np.float32)

    w = torch._sparse_coo_tensor_unsafe(wft_indices, 
            wft_vals, (1, NUM_FEATURES))
    b = torch._sparse_coo_tensor_unsafe(bft_indices, 
            bft_vals, (1, NUM_FEATURES))

    w._coalesced_(True)
    b._coalesced_(True)

    stm = torch.tensor([fts.stm])

    return w, b, stm


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, 4)
        self.l1 = nn.Linear(2 * 4, 8)
        self.l2 = nn.Linear(8, 1)

    def forward(self, wfts, bfts, stm):
        w = self.ft(wfts)
        b = self.ft(bfts)

        accumulator = stm * torch.cat([w, b], dim=1)
        accumulator += (1 - stm) * torch.cat([b, w], dim=1)

        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        return self.l2(l2_x)


def loss_fn(lambda_, pred, w, b, stm, score, outcome):
    target = (score / SCALE).sigmoid()
    return ((pred - target) ** 2).mean()


def train_step(nnue: NNUE, batch: SparseBatch, optimizer):
    w, b, stm, score, result = get_tensors(batch)

    pred = nnue.forward(w, b, stm)
    loss = loss_fn(1.0, pred, w, b, stm, score, result)
    loss.backward()
    optimizer.step()
    nnue.zero_grad()

    return loss.item()


def eval_fen(dll, fen, model):
    fts = dll.get_features(fen.encode('utf-8'))
    w, b, stm = get_entry_tensors(fts.contents)
    with torch.no_grad():
        score = model.forward(w, b, stm).logit(eps=1e-6).item() * SCALE

    dll.destroy_features(fts)
    return score


def main():
    dll = load_dll('./my_dll.dll')
    dataset = SparseBatchDataset(dll, '37540_games.bin')
    num_epochs = 100

    nnue = NNUE()
    # optimizer = torch.optim.Adam(nnue.parameters(), lr=0.01)
    optimizer = Ranger21(nnue.parameters(), lr=0.01, 
            num_epochs=num_epochs, num_batches_per_epoch=348)

    if os.path.isfile('state2.pt'):
        nnue.load_state_dict(torch.load('state2.pt', 
            map_location=torch.device('cpu')))

    n = 0
    for epoch in range(num_epochs):
        mean_loss = 0
        for i, batch in enumerate(dataset):
            assert batch is not None

            loss = train_step(nnue, batch, optimizer)
            mean_loss += loss

            print(f'epoch: {epoch:2}, batch: {i:4}/{n},',
                  f'pos: {BATCH_SIZE * i:8}, loss: {loss:.4f}', end='\r')
        n = i + 1

        mean_loss /= n
        print(f'epoch {epoch}, mean loss {mean_loss:.4f}', ' ' * 64)
        torch.save(nnue.state_dict(), 'state.pt')


if __name__ == '__main__':
    main()

