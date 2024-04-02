import chess

N_FEATURES = 768
N_MAX_ACTIVE_FTS = 32

def piece_to_index(pov, p: chess.Piece):
    p_idx = 2 * (p.piece_type - 1) + int(p.color != pov)
    return p_idx

def feature_index(pov: chess.Color, sq: int, p: chess.Piece):
    p_idx = piece_to_index(pov, p)
    o_sq = sq ^ (0 if pov == chess.WHITE else 56)
    return o_sq + p_idx * 64

def get_active_features(board: chess.Board):
    def piece_features(side):
        return [feature_index(side, sq, p) for sq, p in board.piece_map().items()]
    return piece_features(chess.WHITE), piece_features(chess.BLACK)
