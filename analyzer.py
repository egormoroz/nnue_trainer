import asyncio
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import ctypes
import atexit

class BinWriter:
    def __init__(self, dll_path: str):
        self.dll = ctypes.cdll.LoadLibrary(dll_path)
        atexit.register(self.close_file)

    def open_file(self, path: str) -> bool:
        return bool(self.dll.open_file(path.encode('utf-8')))

    def write_entry(self, fen: str, score: int) -> bool:
        return bool(self.dll.write_entry(fen.encode('utf-8'),
            score))

    def close_file(self) -> bool:
        return bool(self.dll.close_file())


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not (
        move.promotion
        or board.gives_check(move)
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
    )

counter = 0

async def analyze_game(engine, game, limit, 
        duplicates, writer) -> None:
    global counter
    board = game.board()
    for move in game.mainline_moves():
        quiet = is_quiet(board, move)
        board.push(move)

        pos_hash = chess.polyglot.zobrist_hash(board)

        if not quiet or pos_hash in duplicates:
            continue

        duplicates.add(pos_hash)

        info = await engine.analyse(board, limit)

        if not 'score' in info:
            continue

        score = info['score'].relative.score()
        fen = board.fen()

        print(f"{counter}. {score} {fen}")
        writer.write_entry(fen, score)
        counter += 1


async def analyze_pgn(e_path, pgn, limit, writer) -> None:
    transport, engine = await chess.engine.popen_uci(e_path)
    duplicates = set()
    game = chess.pgn.read_game(pgn)
    while game is not None:
        await analyze_game(engine, game, limit, 
            duplicates, writer)
        game = chess.pgn.read_game(pgn)


async def main() -> None:
    pgn = open('battle.pgn')
    writer = BinWriter('./packer.dll')
    writer.open_file('packed.bin')

    limit = chess.engine.Limit(depth=12)
    workers = [
        analyze_pgn('saturn.exe', pgn, limit, writer)
        for _ in range(8)
    ]
    await asyncio.gather(*workers)



asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())

