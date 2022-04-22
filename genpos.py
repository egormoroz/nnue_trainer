import asyncio
import chess
import chess.engine
import chess.pgn
import chess.polyglot

import random
from collections import namedtuple
from enum import IntEnum

from ffi import *

RANDOM_PLIES  = [10, 12, 14, 16]

MAX_PLIES = 400

MIN_DRAW_PLY = 80
DRAW_SCORE = 10
DRAW_COUNT = 10

EVAL_LIMIT = 10000


Position = namedtuple('Position', 'fen score')


class Outcome(IntEnum):
    WHITE_WINS = 0
    BLACK_WINS = 1
    DRAW = 2

    def from_str(s: str):
        if s == '1-0':
            return Outcome.WHITE_WINS
        elif s == '0-1':
            return Outcome.BLACK_WINS
        elif s == '1/2-1/2':
            return Outcome.DRAW
        return None


def write_positions(writer: BinWriter, positions: list[Position], 
        outcome: Outcome) -> None:
    for pos in positions:
        writer.write_entry(pos.fen, pos.score, int(outcome))


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not (
        move.promotion
        or board.gives_check(move)
        or board.is_capture(move)
        or board.is_en_passant(move)
        or board.is_check()
    )


def setup_board() -> chess.Board:
    board = chess.Board()

    for k in range(random.choice(RANDOM_PLIES)):
        moves = list(board.legal_moves)

        if not moves:
            board.pop()
            break
        elif len(moves) == 1:
            idx = 0
        else:
            idx = random.randint(0, len(moves) - 1)
        board.push(moves[idx])
        if board.is_game_over(claim_draw=True):
            board.pop()
            break

    return board


async def play_game(engine_white: chess.engine.UciProtocol,
        engine_black: chess.engine.UciProtocol,
        limit: chess.engine.Limit,
        duplicates: set) -> (list[Position], Outcome):

    positions = []
    board = setup_board()
    outcome = Outcome.DRAW
    draw_count = 0

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            result = await engine_white.play(board, limit,
                    info=chess.engine.Info.SCORE)
        else:
            result = await engine_black.play(board, limit,
                    info=chess.engine.Info.SCORE)
        
        if 'score' not in result.info or not is_quiet(board, result.move):
            board.push(result.move)
            continue

        score = result.info['score']
        score_cp = score.relative.score(mate_score=30000)
        if abs(score_cp) >= EVAL_LIMIT:
            if score.white().score(mate_score=30000) > 0:
                outcome = Outcome.WHITE_WINS
            else:
                outcome = Outcome.BLACK_WINS
            break
        
        phash = chess.polyglot.zobrist_hash(board)
        if phash in duplicates:
            board.push(result.move)
            continue
        duplicates.add(phash)

        positions.append(Position(fen=board.fen(), score=score_cp))

        ply = board.halfmove_clock
        if ply >= MAX_PLIES:
            outcome = 2
            break

        if ply > MIN_DRAW_PLY and abs(score_cp) < DRAW_SCORE:
            draw_count += 1
            if draw_count >= DRAW_COUNT:
                outcome = Outcome.DRAW
                break
        else:
            draw_count = 0

        board.push(result.move)

    if board.is_game_over(claim_draw=True):
        outcome = Outcome.from_str(board.result(claim_draw=True))

    return positions, outcome


class PosCounter:
    def __init__(self, initial_value: int, final_value: int):
        self.counter = initial_value
        self.final_value = final_value

    def done(self) -> bool:
        return self.counter >= self.final_value

    def add(self, n: int) -> None:
        self.counter += n


async def run_session(writer: BinWriter, 
            eng_path: str, counter: PosCounter,
            limit: chess.engine.Limit,
            duplicates: set) -> None:

    _, engine_white = await chess.engine.popen_uci(eng_path)
    _, engine_black = await chess.engine.popen_uci(eng_path)

    while not counter.done():
        positions, outcome = await play_game(engine_white,
            engine_black, limit, duplicates)

        counter.add(len(positions))
        for pos in positions:
            print(pos)
            writer.write_entry(pos.fen, pos.score, int(outcome))

        print(f'[{counter.counter} / {counter.final_value}]')

    await engine_white.quit()
    await engine_black.quit()


async def main() -> None:
    random.seed(0xdeadbeef)
    dll = load_dll('./dataloader.dll')
    writer = BinWriter(dll, 'lul.bin')
    eng_path = 'saturn.exe'

    limit = chess.engine.Limit(time=0.2, depth=16)
    duplicates = set()
    counter = PosCounter(0, 200)

    workers = [
        run_session(writer, eng_path, counter, limit, duplicates)
        for _ in range(1)
    ]
    
    await asyncio.gather(*workers)
    print(f'hash: {writer.get_hash():x}')
    print(len(duplicates))


asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())

