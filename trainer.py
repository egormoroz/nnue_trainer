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

def get_tensors(sb: SparseBatch, device):
    stm = np.ctypeslib.as_array(sb.stm)[:sb.size]
    stm = torch.from_numpy(stm.reshape((sb.size, 1))).to(device)

    score = np.ctypeslib.as_array(sb.score)[:sb.size]
    score = torch.from_numpy(score.reshape((sb.size, 1))).to(device)

    result = np.ctypeslib.as_array(sb.result)[:sb.size]
    result = torch.from_numpy(result.reshape((sb.size, 1))).to(device)

    wft_indices = np.ctypeslib.as_array(sb.wft_indices)
    wft_indices = wft_indices[:sb.n_wfts * 2].reshape((sb.n_wfts, 2)).T
    bft_indices = np.ctypeslib.as_array(sb.bft_indices)
    bft_indices = bft_indices[:sb.n_bfts * 2].reshape((sb.n_bfts, 2)).T

    wft_vals = torch.ones(sb.n_wfts)
    bft_vals = torch.ones(sb.n_bfts)

    w = torch._sparse_coo_tensor_unsafe(wft_indices, 
            wft_vals, (sb.size, NUM_FEATURES)).to(device)
    b = torch._sparse_coo_tensor_unsafe(bft_indices, 
            bft_vals, (sb.size, NUM_FEATURES)).to(device)

    w._coalesced_(True)
    b._coalesced_(True)

    return w, b, stm, score, result


def get_entry_tensors(fts, device):
    wft_indices = np.ctypeslib.as_array(fts.wft_indices)
    wft_indices = wft_indices[:fts.n_wfts].reshape((1, fts.n_wfts))
    wft_indices = np.concatenate((np.zeros_like(wft_indices), wft_indices))

    bft_indices = np.ctypeslib.as_array(fts.bft_indices)
    bft_indices = bft_indices[:fts.n_bfts].reshape((1, fts.n_bfts))
    bft_indices = np.concatenate((np.zeros_like(bft_indices), bft_indices))

    wft_vals = np.ones(fts.n_wfts, dtype=np.float32)
    bft_vals = np.ones(fts.n_bfts, dtype=np.float32)

    w = torch._sparse_coo_tensor_unsafe(wft_indices, 
            wft_vals, (1, NUM_FEATURES)).to(device)
    b = torch._sparse_coo_tensor_unsafe(bft_indices, 
            bft_vals, (1, NUM_FEATURES)).to(device)

    w._coalesced_(True)
    b._coalesced_(True)

    stm = torch.tensor([fts.stm]).to(device)

    return w, b, stm


def crelu(x):
    return torch.clamp(x, 0.0, 1.0)

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, 128)
        self.l1 = nn.Linear(2 * 128, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

        self.weight_clipping = [
            {'params' : [self.ft.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l3.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
        ]

    def clip_weights(self):
        for group in self.weight_clipping:
            for p in group['params']:
                p_data_fp32 = p.data
                min_weight = group['min_weight']
                max_weight = group['max_weight']
                p_data_fp32.clamp_(min_weight, max_weight)
                p.data.copy_(p_data_fp32)

    def forward(self, wfts, bfts, stm):
        w = self.ft(wfts)
        b = self.ft(bfts)

        accumulator = stm * torch.cat([w, b], dim=1)
        accumulator += (1 - stm) * torch.cat([b, w], dim=1)

        x = crelu(accumulator)
        x = crelu(self.l1(x))
        x = crelu(self.l2(x))
        x = self.l3(x)

        return x


def loss_fn(lambda_, pred, w, b, stm, score, outcome):
    target = (score / SCALE).sigmoid()
    return ((pred - target) ** 2).mean()


def train_step(nnue: NNUE, batch: SparseBatch, optimizer, device):
    w, b, stm, score, result = get_tensors(batch, device)

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


def calculate_validation_loss(nnue: NNUE, 
        dataset: SparseBatchDataset, device):
    nnue.eval()
    val_loss = []
    with torch.no_grad():
        for batch in dataset:
            w, b, stm, score, result = get_tensors(batch, device)

            pred = nnue.forward(w, b, stm)
            loss = loss_fn(1.0, pred, w, b, stm, score, result)
            val_loss.append(loss)
        nnue.train()

    return torch.mean(torch.tensor(val_loss))


def main():
    dll = load_dll('./my_dll.dll')
    training_dataset = SparseBatchDataset(dll, 'games.bin')
    validation_dataset = SparseBatchDataset(dll, 'validation.bin')

    num_epochs = 100

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    nnue = NNUE().to(device)
    # optimizer = torch.optim.Adam(nnue.parameters(), lr=0.01)
    optimizer = Ranger21(nnue.parameters(), lr=0.01, 
            num_epochs=num_epochs, num_batches_per_epoch=348)

    saved_path = '256_32_32_32.pt'
    best_val_loss = 1.0
    if os.path.isfile(saved_path):
        try:
            nnue.load_state_dict(torch.load(saved_path))
        except:
            nnue.load_state_dict(torch.load(saved_path,
                map_location=torch.device('cpu')))
        val_loss = calculate_validation_loss(nnue, validation_dataset, device)
        best_val_loss = val_loss
        print(f'validation loss {val_loss:.4f}')

    n = 0
    for epoch in range(num_epochs):
        train_loss = 0
        for i, batch in enumerate(training_dataset):
            loss = train_step(nnue, batch, optimizer, device)
            train_loss += loss

            print(f'epoch: {epoch:2}, batch: {i:4}/{n},',
                  f'pos: {BATCH_SIZE * i:8}, loss: {loss:.4f}', end='\r')
        n = i + 1

        train_loss /= n
        val_loss = calculate_validation_loss(nnue, validation_dataset, device)

        printf(f'epoch {epoch}, training loss {train_loss:.4f}',
                f'validation loss {val_loss:.4f}', ' ' * 64)

        if val_loss < best_val_loss:
            torch.save(nnue.state_dict(), saved_path)


if __name__ == '__main__':
    main()

