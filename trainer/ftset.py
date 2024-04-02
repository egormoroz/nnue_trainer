import chess


N_KING_BUCKETS = 4
N_FEATURES = 12*64*N_KING_BUCKETS
N_MAX_ACTIVE_FTS = 32

KING_BUCKETS = [
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3,
]

def piece_to_index(pov, p: chess.Piece):
    p_idx = 2 * (p.piece_type - 1) + int(p.color != pov)
    return p_idx

def feature_index(pov: chess.Color, sq: int, p: chess.Piece, ksq: int):
    p_idx = piece_to_index(pov, p)
    o_sq = sq ^ (0 if pov == chess.WHITE else 56)
    o_ksq = ksq ^ (0 if pov == chess.WHITE else 56)

    return o_sq + 64*p_idx + 64*12*KING_BUCKETS[o_ksq]

def get_active_features(board: chess.Board):
    def piece_features(side):
        ksq = board.king(side) 
        assert ksq is not None

        return [feature_index(side, sq, p, ksq) for sq, p in board.piece_map().items()]
    return piece_features(chess.WHITE), piece_features(chess.BLACK)
