import torch
import torch.nn as nn

from transformer import FeatureTransformer
from halfkp import *

import chess

from ranger21 import Ranger21

WDL_SCALE = 162

S_W = 64
S_A = 127 # flaot acts are 0..1 => int8 acts are 0..127

# unsure how to choose this value
# it should be choosen s.t. the values in the last layer aren't clipped too much?..

# model_new.pt has S_O = 128
# S_O = 128 

# let's try a bigger value since the net struggles to get scores > 500
S_O = 256

class Model(nn.Module):
    def __init__(self, n_fts):
        super().__init__()
        self.ft = FeatureTransformer(n_fts, 256 + 1)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_out = nn.Linear(32, 1)

        self._init_psqt()

    @torch.no_grad()
    def _init_psqt(self):
        p_vals = [0, 82, 337, 365, 477, 1025,  0]
        for pt in range(chess.PAWN, chess.KING):
            p_idx = 2 * pt
            p_value = p_vals[pt] / S_O
            for psq in range(N_SQ):
                for ksq in range(N_SQ):
                    i_real = psq + N_SQ * p_idx + N_SQ*N_PC * ksq
                    self.ft.weight.data[i_real, -1] = p_value

    @torch.no_grad()
    def coalesce_transformer(self):
        self.ft.coalesce_weights()

    @torch.no_grad()
    def _clip_weights(self):
        self.fc1.weight.data.clip_(-127/S_W, 127/S_W)
        self.fc2.weight.data.clip_(-127/S_W, 127/S_W)
        self.fc_out.weight.data.clip_(-127*S_A/(S_W*S_O), 127*S_A/(S_W*S_O))

    def forward(self, wft_ics, wft_vals, bft_ics, bft_vals, stm):
        wfts = self.ft(wft_ics, wft_vals)
        bfts = self.ft(bft_ics, bft_vals)

        wfts, wpsqt = torch.split(wfts, 256, dim=1)
        bfts, bpsqt = torch.split(bfts, 256, dim=1)

        x = (1 - stm) * torch.cat((wfts, bfts), dim=-1)
        x += stm * torch.cat((bfts, wfts), dim=-1)

        x = torch.clip(x, 0, 1)
        x = torch.clip(self.fc1(x), 0, 1)
        x = torch.clip(self.fc2(x), 0, 1)
        x = self.fc_out(x)

        x = x + (wpsqt - bpsqt) * (0.5 - stm)
        return x

    @torch.no_grad()
    def evaluate_board(self, board: chess.Board):
        wft_ics, bft_ics = get_active_features(board)
        wft_ics += [-1] * (MAX_REAL_ACTIVE_FTS - len(wft_ics))
        bft_ics += [-1] * (MAX_REAL_ACTIVE_FTS - len(bft_ics))

        device = next(self.parameters()).device
        wft_ics = torch.tensor(wft_ics, dtype=torch.int32, device=device).view(1, -1)
        bft_ics = torch.tensor(bft_ics, dtype=torch.int32, device=device).view(1, -1)

        wft_vals = torch.ones((1, MAX_REAL_ACTIVE_FTS), dtype=torch.float32, device=device)
        bft_vals = torch.ones((1, MAX_REAL_ACTIVE_FTS), dtype=torch.float32, device=device)

        stm = [0 if board.turn == chess.WHITE else 1]
        stm = torch.tensor(stm, dtype=torch.float32, device=device).view(1, 1)

        pred = self.forward(wft_ics, wft_vals, bft_ics, bft_vals, stm)
        return pred * S_O

    def configure_optimizers(self, config):
        # opt = torch.optim.AdamW(self.parameters())
        opt = Ranger21(self.parameters(), lr=config.max_lr, warmdown_min_lr=config.min_lr,
                       num_epochs=config.n_epochs, 
                       num_batches_per_epoch=config.n_batches_per_epoch,
                       warmdown_active=False, use_warmup=False)
        return opt


def compute_loss(pred, score, game_result, lambda_):
    wdl_target = torch.sigmoid(score / WDL_SCALE)
    wdl_target = lambda_ * wdl_target + (1 - lambda_) * game_result

    wdl_pred = torch.sigmoid(pred * S_O / WDL_SCALE)
    loss = torch.pow(wdl_pred - wdl_target, 2).mean()
    return loss
