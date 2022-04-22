from ffi import *
from trainer import *

dll = load_dll('./dataloader.dll')
if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

nnue = NNUE()
nnue.eval()

def load_params():
    nnue.load_state_dict(torch.load('halfkav2.pt', map_location=dev))

def get_tensors(fen):
    fts = dll.get_features(fen.encode('utf-8'))
    w, b, stm = get_entry_tensors(fts.contents, dev)
    dll.destroy_features(fts)
    return w, b, stm

def ev_fen(fen):
    with torch.no_grad():
        return nnue.forward(*get_tensors(fen))

def ev_psqt(fen):
    wfts, bfts, stm = get_tensors(fen)
    with torch.no_grad():
        wp = nnue.ft(wfts)
        bp = nnue.ft(bfts)

        w, wpsqt = torch.split(wp, wp.shape[1]-1, dim=1)
        b, bpsqt = torch.split(bp, bp.shape[1]-1, dim=1)

        return (wpsqt + bpsqt) * (stm - 0.5)


