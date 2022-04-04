from ffi import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NUM_FEATURES = 40960

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


def loss_fn(lambda_, pred, w, b, stm, score, result):
    epsilon = 1e-12
    scaling = 500
    p = (pred / scaling).sigmoid()
    wdl_eval_target = (score / scaling).sigmoid()
    t = lambda_ * wdl_eval_target + (1 - lambda_) * result

    loss = (t * (t + epsilon).log() + (1 - t) * (1 - t + epsilon).log()) \
          -(t * (p + epsilon).log() + (1 - t) * (1 - p + epsilon).log())
    return loss.mean()


def loss_function(lambda_, pred, w, b, stm, score, outcome):
    q = pred
    t = outcome
    p = (score / 400.0).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    return loss

def train_step(nnue: NNUE, batch: SparseBatch, optimizer):
    w, b, stm, score, result = get_tensors(batch)

    pred = nnue.forward(w, b, stm)
    loss = loss_function(0.0, pred, w, b, stm, score, result)
    loss.backward()
    optimizer.step()
    nnue.zero_grad()
    
    print(f'loss: {loss.item()}')


def main():
    dll = load_dll('./my_dll.dll')
    r = BinReader(dll, 'out.bin')

    nnue = NNUE()
    optimizer = torch.optim.Adam(nnue.parameters())

    batch = r.get_batch()
    # batches = []

    while r.next_batch() != 0:
        train_step(nnue, batch, optimizer)

        # batch_copy = SparseBatch()
        # ct.memmove(ct.addressof(batch_copy), 
        #     ct.addressof(batch), ct.sizeof(batch))
        # batches.append(batch_copy)

    # for _ in range(10_000):
    #     for b in batches:
    #         train_step(nnue, b, optimizer)
    

if __name__ == '__main__':
    main()

