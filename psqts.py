import chess

def halfka_idx(ksq: int, psq: int, p: chess.Piece):
    p_idx = (p.piece_type - 1) * 2 + int(not p.color)
    return (ksq * 64 + psq) * 10 + p_idx


def halfka_psqts():
  piece_values = {
    chess.PAWN : 126,
    chess.KNIGHT : 781,
    chess.BISHOP : 825,
    chess.ROOK : 1276,
    chess.QUEEN : 2538
  }

  values = [0] * 40960

  for ksq in range(64):
    for s in range(64):
      for pt, val in piece_values.items():
        idxw = halfka_idx(ksq, s, chess.Piece(pt, chess.WHITE))
        idxb = halfka_idx(ksq, s, chess.Piece(pt, chess.BLACK))
        values[idxw] = val
        values[idxb] = -val

  return values

