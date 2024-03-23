import chess
import torch

N_SQ = 64
N_PC = 10
N_KING_BUCKETS = N_SQ // 2

N_FT = N_SQ * N_PC * N_KING_BUCKETS

N_VIRT_FT = N_SQ * N_PC

MAX_REAL_ACTIVE_FTS = 30
MAX_VIRT_ACTIVE_FTS = 30

KING_BUCKETS = [
    -1, -1, -1, -1,  3,  2,  1,  0,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 31, 30, 29, 28,
]


def orient(pov: chess.Color, sq: int, ksq: int):
    file = ksq % 8
    hor_flip = 7 * (file < 4)
    vert_flip = 0 if pov == chess.WHITE else 56
    return sq ^ hor_flip ^ vert_flip

def piece_to_index(pov, p: chess.Piece):
    p_idx = 2 * (p.piece_type - 1) + int(p.color != pov)
    return p_idx

def halfkp_idx(pov: chess.Color, ksq: int, psq: int, p: chess.Piece):
    p_idx = piece_to_index(pov, p)
    o_ksq = orient(pov, ksq, ksq)
    o_sq = orient(pov, psq, ksq)
    return o_sq + N_SQ * p_idx + N_SQ*N_PC * KING_BUCKETS[o_ksq]


def get_active_features(board: chess.Board):
    def piece_features(side):
        indices = []
        ksq = board.king(side)
        assert ksq
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
            o_sq = orient(side, sq, ksq)
            indices.append(N_FT + o_sq + N_SQ * p_idx)
        return indices
    return piece_features(chess.WHITE), piece_features(chess.BLACK)


@torch.no_grad()
def coalesce_real_virtual_weights(combined: torch.Tensor):
    w = combined[:N_FT, :].clone()
    for psq in range(N_SQ):
        for p_idx in range(N_PC):
            i_virt = N_FT + psq + N_SQ * p_idx
            for ksq in range(N_KING_BUCKETS):
                i_real = psq + N_SQ * p_idx + N_SQ*N_PC * ksq
                w[i_real, :] += combined[i_virt, :]

    return w

