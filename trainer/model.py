import torch
import torch.nn as nn

import chess

from transformer import FeatureTransformer
from ftset import *
from ranger21 import Ranger21

WDL_SCALE = 162

FT_WEIGHT_MAXABS = 910

FC_OUT_MAXABS = 128
S_W = 64
S_A = 255
S_O = 400

N_HIDDEN = 512


class PSQT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_FEATURES + 1, 1, padding_idx=0)
        self._init_psqt()

    @torch.no_grad()
    def _init_psqt(self):
        piece_val = {
            chess.PAWN: 82,
            chess.KNIGHT: 337,
            chess.BISHOP: 365,
            chess.ROOK: 477,
            chess.QUEEN: 1025,
        }

        self.emb.weight.data.zero_()

        for ksq in range(64):
            for pt, value in piece_val.items():
                value = value / S_O
                for sq in range(64):
                    wp, bp = [chess.Piece(pt, c) for c in chess.COLORS]
                    widx = feature_index(chess.WHITE, sq, wp, ksq)
                    bidx = feature_index(chess.WHITE, sq, bp, ksq)
                    self.emb.weight.data[1 + widx] = value
                    self.emb.weight.data[1 + bidx] = -value

    def forward(self, ics):
        x = self.emb(ics + 1).sum(dim=(1, 2))
        return x.view(-1, 1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = FeatureTransformer(N_FEATURES, N_HIDDEN)
        self.psqt = PSQT()
        self.fc_out = nn.Linear(2*N_HIDDEN, 1)

    def forward(self, wft_ics, bft_ics, stm):
        wfts, bfts = self.ft(wft_ics), self.ft(bft_ics)
        wpsqt, bpsqt = self.psqt(wft_ics), self.psqt(bft_ics)

        x = (1 - stm) * torch.cat((wfts, bfts), dim=-1)
        x += stm * torch.cat((bfts, wfts), dim=-1)

        x = self.fc_out(torch.clip(x, 0, 1).pow(2))

        return x + (wpsqt - bpsqt) * (0.5 - stm)

    @torch.no_grad()
    def _clip_weights(self):
        self.ft.weight.data.clip(-FT_WEIGHT_MAXABS/S_A, FT_WEIGHT_MAXABS/S_A)
        self.fc_out.weight.data.clip(-FC_OUT_MAXABS/S_W, FC_OUT_MAXABS/S_W)

    def configure_optimizers(self, config):
        opt = Ranger21(self.parameters(), lr=config.max_lr, warmdown_min_lr=config.min_lr,
                       num_epochs=config.n_epochs, 
                       num_batches_per_epoch=config.n_batches_per_epoch,
                       warmdown_active=False, use_warmup=False)
        return opt

    @torch.no_grad()
    def evaluate_board(self, board: chess.Board):
        wft_ics, bft_ics = get_active_features(board)
        wft_ics += [-1] * (N_MAX_ACTIVE_FTS - len(wft_ics))
        bft_ics += [-1] * (N_MAX_ACTIVE_FTS - len(bft_ics))

        device = next(self.parameters()).device
        wft_ics = torch.tensor(wft_ics, dtype=torch.int32, device=device).view(1, -1)
        bft_ics = torch.tensor(bft_ics, dtype=torch.int32, device=device).view(1, -1)

        stm = [0 if board.turn == chess.WHITE else 1]
        stm = torch.tensor(stm, dtype=torch.float32, device=device).view(1, 1)

        pred = self.forward(wft_ics, bft_ics, stm)
        return pred * S_O


def compute_loss(pred, score, game_result, lambda_):
    wdl_target = torch.sigmoid(score / WDL_SCALE)
    wdl_target = lambda_ * wdl_target + (1 - lambda_) * game_result

    wdl_pred = torch.sigmoid(pred * S_O / WDL_SCALE)
    loss = torch.pow(wdl_pred - wdl_target, 2).mean()
    return loss

