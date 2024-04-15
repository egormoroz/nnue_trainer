import torch
from model import *
import fire
from pathlib import Path
import ffi
import os


def write_transformer(buf: bytearray, transformer):
    bias = transformer.bias.data.ravel()
    bias = bias.mul(S_A).round().to(torch.int16)
    buf.extend(bias.numpy().tobytes())
    
    weight = transformer.weight.data.ravel()
    weight = weight.mul(S_A).round().to(torch.int16)
    buf.extend(weight.numpy().tobytes())


def serialize(buf: bytearray, model: Model):
    psqt = model.psqt.emb.weight.data.ravel()[1:]
    psqt = psqt.mul(S_A).round().to(torch.int16)
    buf.extend(psqt.numpy().tobytes())

    write_transformer(buf, model.ft)

    bias = model.fc_out.bias.data.ravel()
    bias = bias.mul(S_A * S_W).round().to(torch.int16)
    buf.extend(bias.numpy().tobytes())
    
    weight = model.fc_out.weight.data.ravel()
    weight = weight.mul(S_W).round().to(torch.int16)
    buf.extend(weight.numpy().tobytes())

def serialize_net(pt_path_in: str, nnue_path_out: str, compress=False):
    model = Model()
    model.load_state_dict(torch.load(pt_path_in))

    buf = bytearray()
    serialize(buf, model)
    with open(nnue_path_out, 'wb') as f:
        f.write(buf)
    
    if compress:
        ffi.compress_net(
            nnue_path_out, 
            nnue_path_out + '.packed',
        )
        os.rename(nnue_path_out + '.packed', nnue_path_out)

def serialize_cli(*pt_files):
    for pt_in in pt_files:
        path = Path(pt_in)
        folder, name = path.parent.absolute(), path.stem
        out_path = f'{folder}/{name}.nnue'
        
        print(pt_in, '->', out_path)
        serialize_net(pt_in, out_path)


if __name__ == '__main__':
    fire.Fire(serialize_cli)
