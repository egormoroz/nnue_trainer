from model import Model, compute_loss
import ffi
from halfkp import *

import os
import torch
import math
from tqdm.auto import trange
import dataclasses
import fire

import wandb

from testnets import NetTester, TestConfig

torch.set_float32_matmul_precision('high')


class EMA:
    def __init__(self, initial=None, k=0.1):
        self.value = initial
        self.k = k

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = (1 - self.k) * self.value + self.k * x


@dataclasses.dataclass
class Config:
    experiment_name: str
    train_packname: str
    val_packname: str

    resume_from: str

    save_every: int

    min_lr: float
    max_lr: float

    # how much of the first epoch should be a warm up
    p_warmup: float

    n_batches_per_epoch: int
    n_epochs: int

    use_factorizer: bool
    wdl_lambda: float

    # dataloader config
    batch_size: int
    n_prefetch: int
    n_workers: int

    n_restarts: int


def lerp(t, a, b):
    return (1 - t) * a + t * b


def exerp(t, a, b, gamma=1.0):
    assert a > 0 and b > 0
    alpha, beta = a, math.log(b / a)
    return alpha * math.exp(beta * t**gamma)

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

        self.on_epoch_end = None

        self.train_stream = ffi.BatchStream(
            config.train_packname,
            n_prefetch=config.n_prefetch,
            n_workers=config.n_workers,
            batch_size=config.batch_size,
            add_virtual=config.use_factorizer,
            wait_on_end=False,
        )

        self.val_stream = ffi.BatchStream(
            config.val_packname,
            n_prefetch=config.n_prefetch,
            n_workers=config.n_workers,
            batch_size=config.batch_size,
            add_virtual=config.use_factorizer,
            wait_on_end=False,
        )


    def get_lr(self, t, max_t, eta_min, eta_max):
        return eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(t / max_t * math.pi))

    @torch.no_grad()
    def compute_val_loss(self):
        assert self.model

        cfg = self.config
        device = next(self.model.parameters()).device
        training = self.model.training
        self.model.eval()

        cum_loss, n = 0, 0
        for _ in range(100):
            batch = self.val_stream.next_batch()
            assert batch

            wft_ics, bft_ics, stm, score, result = batch.to_torch(device)

            pred = self.model(wft_ics, bft_ics, stm)
            loss = compute_loss(pred, score, result, cfg.wdl_lambda)
            cum_loss += loss.item()
            n += 1

        self.model.train(training)
        return cum_loss / max(1, n)


    def train(self, model: Model):
        opt = model.configure_optimizers(self.config)
        cfg = self.config
        device = next(model.parameters()).device
        self.model = model

        batch_loss = EMA()

        train_loss = 0
        val_loss = 0

        restart_idx = 0
        step = 0
        total_steps = cfg.n_batches_per_epoch * cfg.n_epochs

        total_i = total_steps / (2**cfg.n_restarts - 1)
        total_i = int(math.ceil(total_i))

        eta_min, eta_max = cfg.min_lr, cfg.max_lr

        warmup_steps = int(cfg.p_warmup * cfg.n_batches_per_epoch)

        for epoch in trange(cfg.n_epochs):
            cum_loss = 0
            for batch_idx in (t := trange(cfg.n_batches_per_epoch)):
                batch = self.train_stream.next_batch()
                assert batch

                if epoch == 0 and batch_idx < warmup_steps:
                    lr = lerp(batch_idx/warmup_steps, cfg.min_lr, cfg.max_lr)
                else:
                    lr = self.get_lr(step, total_i, eta_min, eta_max)
                    step += 1

                if step >= total_i:
                    restart_idx += 1
                    step = 0
                    total_i *= 2
                    abs_step = epoch * cfg.n_batches_per_epoch + batch_idx
                    total_i = min(total_i, total_steps - abs_step + 1)
                    
                    # r = abs_step / total_steps
                    r = restart_idx / cfg.n_restarts
                    eta_min = exerp(r, cfg.min_lr, cfg.min_lr / 10, 1.6)
                    eta_max = exerp(r, cfg.max_lr, cfg.max_lr / 10, 1.6)

                for g in opt.param_groups:
                    g['lr'] = lr

                wft_ics, bft_ics, stm, score, result = batch.to_torch(device)
                pred = model(wft_ics, bft_ics, stm)
                loss = compute_loss(pred, score, result, cfg.wdl_lambda)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model._clip_weights()

                cum_loss += loss.item()
                batch_loss.update(loss.item())

                t.set_description('epoch {} BL {:.5f} TL {:.5f} VL {:.5f} LR {:.2e}'.format(
                    epoch, batch_loss.value, train_loss, val_loss, lr))

            train_loss = cum_loss / cfg.n_batches_per_epoch
            val_loss = self.compute_val_loss()

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, train_loss, val_loss)


def train(experiment_name, train_packname, val_packname, save_every=10,
          min_lr=1e-5, max_lr=8e-4, p_warmup=0.5, n_batches_per_epoch=6000, 
          n_epochs=200, wdl_lambda=1.0, batch_size=16384,
          n_prefetch=4, n_workers=4, resume_from='', n_restarts=4,
          use_factorizer=True):

    config = Config(
        experiment_name=experiment_name, 
        train_packname=train_packname, val_packname=val_packname,
        save_every=save_every, min_lr=min_lr, max_lr=max_lr, p_warmup=p_warmup,
        n_batches_per_epoch=n_batches_per_epoch, n_epochs=n_epochs, 
        wdl_lambda=wdl_lambda, batch_size=batch_size,
        n_prefetch=n_prefetch, n_workers=n_workers,
        resume_from=resume_from,
        n_restarts=n_restarts,
        use_factorizer=use_factorizer
    )

    test_cfg = TestConfig(
        exp_name=experiment_name,
        cc_cmd='cutechess-cli',
        eng_cmd=f'./saturn',
        round_pairs=1000,
        openings='/home/ktnkdoomer/Documents/openings/UHO_Lichess_4852_v1.epd',
        tc_nodes=30000,
        concurrency=12
    )
    tester = NetTester(test_cfg)

    os.makedirs(experiment_name, exist_ok=True)

    # ffi.load_module('../build/Release/satpymod.dll')
    ffi.load_module('../build/libsatpymod.so')

    model = Model(N_FT + N_VIRT_FT * use_factorizer).cuda()
    if resume_from:
        print(model.load_state_dict(torch.load(resume_from)))
    model.compile()

    wandb.login()
    run = wandb.init(
        project='nnue',
        name=config.experiment_name,
        config=dataclasses.asdict(config),
    )
    assert run

    best_val_loss = float('+inf')
    all_results = []

    def on_epoch_end(epoch, train_loss, val_loss):
        nonlocal best_val_loss
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        results = tester.get_results()
        all_results.extend(results)
        all_results.sort(key=lambda x: x[1], reverse=True)


        if results:
            print()
            for name, elo, err in results:
                print('{} {} +/- {}'.format(name, elo, err))
                wandb.log({ 'elo_diff': elo })

            print('Top 3 nets:')
            for name, elo, err in all_results[:3]:
                print('{} {} +/- {}'.format(name, elo, err))

        if epoch % config.save_every == 0 or epoch + 1 == config.n_epochs:

            name = f'net_{epoch}.pt'
            path = f'{config.experiment_name}/{name}'
            torch.save(model.state_dict(), path)

            tester.enqueue_net(f'net_{epoch}')
            # run.log_model(path, name=name)

    trainer = Trainer(config)
    trainer.on_epoch_end = on_epoch_end # pyright: ignore
    trainer.train(model)


if __name__ == '__main__':
    fire.Fire(train)

