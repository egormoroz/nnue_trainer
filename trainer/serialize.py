import torch
from model import *
import fire
from pathlib import Path


def write_transformer(buf: bytearray, transformer):
    psqt = transformer.weight.data[:, -1].ravel()
    psqt = psqt.mul(S_A).round().to(torch.int16)
    buf.extend(psqt.numpy().tobytes())

    bias = transformer.bias.data[:-1].ravel()
    bias = bias.mul(S_A).round().to(torch.int16)
    buf.extend(bias.numpy().tobytes())
    
    weight = transformer.weight.data[:, :-1].ravel()
    weight = weight.mul(S_A).round().to(torch.int16)
    buf.extend(weight.numpy().tobytes())


def serialize(buf: bytearray, model: Model):
    write_transformer(buf, model.ft)

    bias = model.fc_out.bias.data.ravel()
    bias = bias.mul(S_A * S_W).round().to(torch.int16)
    buf.extend(bias.numpy().tobytes())
    
    weight = model.fc_out.weight.data.ravel()
    weight = weight.mul(S_W).round().to(torch.int16)
    buf.extend(weight.numpy().tobytes())

def serialize_net(pt_path_in, nnue_path_out):
    model = Model()
    model.load_state_dict(torch.load(pt_path_in))

    buf = bytearray()
    serialize(buf, model)
    with open(nnue_path_out, 'wb') as f:
        f.write(buf)

def serialize_cli(*pt_files):
    for pt_in in pt_files:
        path = Path(pt_in)
        folder, name = path.parent.absolute(), path.stem
        out_path = f'{folder}/{name}.nnue'
        
        print(pt_in, '->', out_path)
        serialize_net(pt_in, out_path)


if __name__ == '__main__':
    fire.Fire(serialize_cli)
