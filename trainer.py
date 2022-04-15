import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from ranger21 import Ranger21

from ffi import *
from dataset import *
from model import *

SCALE = 500.0
NNUE_SCALING = 150

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


def loss_fn(lambda_, pred, w, b, stm, score, outcome):
    p = (pred * NNUE_SCALING / SCALE).sigmoid()
    q = (score / SCALE).sigmoid()
    return torch.pow(torch.abs(p - q), 2.6).mean()


def train_step(nnue: NNUE, batch: SparseBatch, optimizer, device):
    w, b, stm, score, result = get_tensors(batch, device)

    pred = nnue.forward(w, b, stm)
    loss = loss_fn(1.0, pred, w, b, stm, score, result)
    loss.backward()
    optimizer.step()
    nnue.zero_grad()

    with torch.no_grad():
        nnue.clip_weights()

    return loss.item()


def eval_fen(dll, fen, model, device):
    fts = dll.get_features(fen.encode('utf-8'))
    w, b, stm = get_entry_tensors(fts.contents, device)
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

    num_epochs = 2

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    nnue = NNUE().to(device)
    # optimizer = torch.optim.Adam(nnue.parameters(), lr=0.01)
    optimizer = Ranger21(nnue.parameters(), lr=0.01, 
            num_epochs=num_epochs, num_batches_per_epoch=348)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=1e-6)

    saved_path = 'asdf.pt'
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
        scheduler.step(val_loss)

        print(f'epoch {epoch}, training loss {train_loss:.4f}',
                f'validation loss {val_loss:.4f}', ' ' * 64)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(nnue.state_dict(), saved_path)


if __name__ == '__main__':
    main()

