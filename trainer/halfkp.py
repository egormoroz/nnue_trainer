import chess
import torch

N_PC = 10
N_SQ = 64
N_FT = N_SQ * N_PC * N_SQ

N_VIRT_FT = N_SQ * N_PC
MAX_REAL_ACTIVE_FTS = 30
MAX_VIRT_ACTIVE_FTS = 30


def orient(pov: chess.Color, sq: int):
    return sq if pov == chess.WHITE else sq ^ 0x3f

def piece_to_index(pov, p: chess.Piece):
    p_idx = 2 * (p.piece_type - 1) + int(p.color != pov)
    return p_idx

def halfkp_idx(pov: chess.Color, ksq: int, psq: int, p: chess.Piece):
    p_idx = piece_to_index(pov, p)
    return orient(pov, psq) + N_SQ * p_idx + N_SQ * N_PC * ksq


def get_active_features(board: chess.Board):
    def piece_features(side):
        indices = []
        ksq = board.king(side)
        assert ksq
        ksq = orient(side, ksq)
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices.append(halfkp_idx(side, ksq, sq, p))
        return indices
    return piece_features(chess.WHITE), piece_features(chess.BLACK)


def get_virtual_active_features(board: chess.Board):
    def piece_features(side):
        ksq = board.king(side)
        assert ksq
        indices = []
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            p_idx = piece_to_index(side, p)
            indices.append(N_FT + orient(side, sq) + N_SQ * p_idx)
        return indices
    return piece_features(chess.WHITE), piece_features(chess.BLACK)


@torch.no_grad()
def coalesce_real_virtual_weights(combined: torch.Tensor):
    w = combined[:N_FT, :].clone()
    for psq in range(N_SQ):
        for p_idx in range(N_PC):
            i_virt = N_FT + psq + N_SQ * p_idx
            for ksq in range(N_SQ):
                i_real = psq + N_SQ * p_idx + N_SQ*N_PC * ksq
                w[i_real, :] += combined[i_virt, :]

    return w

