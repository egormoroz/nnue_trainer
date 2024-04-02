import dataclasses
import subprocess
from pathlib import Path
from serialize import serialize_net

import threading
import time


@dataclasses.dataclass
class TestConfig:
    exp_name: str
    
    cc_cmd: str
    eng_cmd: str

    round_pairs: int
    openings: str

    tc_nodes: int

    concurrency: int

    pgn_out: bool = False


def play_games(net_name, cfg: TestConfig, reserialize=True):
    op_format = 'epd' if cfg.openings.endswith('epd') else 'pgn'
    pt_path = Path('{}/{}.pt'.format(cfg.exp_name, net_name)).absolute()
    net_path = Path('{}/{}.nnue'.format(cfg.exp_name, net_name)).absolute()

    if reserialize or not net_path.exists():
        serialize_net(pt_path, net_path)

    cc_with_args = [
        cfg.cc_cmd,
        '-tournament', 'gauntlet',
        '-concurrency', f'{cfg.concurrency}',
        '-draw', 'movenumber=40', 'movecount=4', 'score=8',
        '-resign', 'movecount=4', 'score=400',
        '-each', 'option.Hash=16', 'proto=uci', 'tc=inf', f'nodes={cfg.tc_nodes}',

        '-openings', f'file={cfg.openings}', f'format={op_format}', 
        'policy=round', 'order=random',

        '-repeat', '-rounds', f'{cfg.round_pairs}', '-games', '2',

    ]

    if cfg.pgn_out:
        cc_with_args += ['-pgnout', f'{cfg.exp_name}.pgn']

    eng_folder = Path(cfg.eng_cmd).readlink().parent.absolute()

    cc_with_args += [
        '-engine', f'cmd={cfg.eng_cmd}', f'dir={eng_folder}', f'name={net_name}',
        f'option.evalfile={net_path}'
    ]

    cc_with_args += [
        '-engine', f'cmd={cfg.eng_cmd}', f'dir={eng_folder}', 'name=base'
    ]

    output = subprocess.check_output(cc_with_args).decode('utf-8')
    elo_diff = output.split('Elo difference', 1)[1].split(':', 1)[1].split(',', 1)[0]
    elo_diff, err_margin = map(float, elo_diff.split('+/-', 1))

    return elo_diff, err_margin


class NetTester:
    def __init__(self, cfg: TestConfig):
        self.cfg = cfg
        self.counter = 0

        self.done_counter = 0
        self.done = []
        self.done_lock = threading.Lock()

        self.threads = []

    def enqueue_net(self, name):
        def session():
            elo_diff, err_margin = play_games(name, self.cfg)
            with self.done_lock:
                self.done.append((name, elo_diff, err_margin))
                self.done_counter += 1

        self.counter += 1
        self.threads.append(threading.Thread(target=session))
        self.threads[-1].start()

    def get_results(self):
        with self.done_lock:
            done = self.done
            self.done = []
            return done

    def is_done(self):
        with self.done_lock:
            return self.done_counter == self.counter

    def join(self):
        for th in self.threads:
            th.join()


if __name__ == '__main__':
    cfg = TestConfig(
        exp_name='net768-512-kingfix',
        cc_cmd='cutechess-cli',
        eng_cmd=f'./saturn',
        round_pairs=20,
        openings='/home/ktnkdoomer/Documents/openings/UHO_Lichess_4852_v1.epd',
        tc_nodes=30000,
        concurrency=12
    )

    tester = NetTester(cfg)

    tester.enqueue_net('net_99')

    while not tester.is_done():
        time.sleep(1)
        results = tester.get_results()
        if results:
            print(results)


