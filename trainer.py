from ffi import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from ranger21 import Ranger21

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
    # q = pred
    # t = outcome
    # p = (score / SCALE).sigmoid()
    # epsilon = 1e-12
    # teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    # outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    # teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    # outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    # result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    # entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    # loss = result.mean() - entropy.mean()
    # return loss
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
    # nnue = NNUE()
    # dll = load_dll('./my_dll.dll')
    # fts = dll.get_features('6k1/2R2p1p/6p1/1P6/8/8/2P2KPP/1r6 w - - 5 36'.encode('utf-8'))
    # w, b, stm = get_entry_tensors(fts.contents)
    # nnue.load_state_dict(torch.load('state.pt'))

    # with torch.no_grad():
    #     score = nnue.forward(w, b, stm).logit().item() * SCALE
    # print(score)


    dll = load_dll('./my_dll.dll')
    r = BinReader(dll, '37540_games.bin')

    nnue = NNUE()
    # optimizer = torch.optim.Adam(nnue.parameters(), lr=0.01)
    optimizer = Ranger21(nnue.parameters(), lr=0.001, 
            num_epochs=100, num_batches_per_epoch=84)

    if os.path.isfile('state.pt'):
        nnue.load_state_dict(torch.load('state.pt'))

    batch = r.get_batch()
    n = r.next_batch()
    nb_batches = 0
    epoch = 0
    mean_loss = 0
    while n != 0 and epoch < 100:
        loss = train_step(nnue, batch, optimizer)
        mean_loss += loss

        if n % 10 == 0:
            print(f'epoch: {epoch:2}, batch: {n:4}/{nb_batches},',
                  f'pos: {BATCH_SIZE * n:8}, loss: {loss:.4f}', end='\r')

        old_n = n
        n = r.next_batch()
        if n < old_n:
            mean_loss /= old_n
            print(f'epoch {epoch}, mean loss {mean_loss:.4f}', ' ' * 64)

            mean_loss = 0
            nb_batches = old_n
            epoch += 1
            torch.save(nnue.state_dict(), 'state.pt')
    

if __name__ == '__main__':
    main()

