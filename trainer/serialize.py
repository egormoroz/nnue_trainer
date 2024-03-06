import torch
from model import *
import fire
import halfkp


def write_transformer(buf: bytearray, transformer):
    bias = transformer.bias.data.ravel()
    bias = bias.mul(127).round().to(torch.int16)
    buf.extend(bias.numpy().tobytes())
    
    weight = transformer.weight.data.ravel()
    weight = weight.mul(127).round().to(torch.int16)
    buf.extend(weight.numpy().tobytes())


def write_layer(buf: bytearray, layer, output: bool = False):
    if output:
        s_b = S_O * S_W
        s_w = S_W * S_O / S_A
    else:
        s_b = S_A * S_W
        s_w = S_W
    bias = layer.bias.data.ravel()
    bias = bias.mul(s_b).round().to(torch.int32)
    buf.extend(bias.numpy().tobytes())
    
    weight = layer.weight.data.ravel()
    weight = weight.mul(s_w).round().to(torch.int8)
    buf.extend(weight.numpy().tobytes())


def serialize(buf: bytearray, model: Model):
    write_transformer(buf, model.ft)
    write_layer(buf, model.fc1)
    write_layer(buf, model.fc2)
    write_layer(buf, model.fc_out, output=True)


def serialize_cli(pt_path_in, nnue_path_out):
    d = torch.load(pt_path_in, map_location=torch.device('cpu'))
    n_ft = d['ft.weight'].shape[0]
    model = Model(n_ft)
    model.load_state_dict(d)

    if n_ft != halfkp.N_FT:
        model.coalesce_transformer()

    buf = bytearray()
    serialize(buf, model)
    with open(nnue_path_out, 'wb') as f:
        f.write(buf)


if __name__ == '__main__':
    fire.Fire(serialize_cli)
