from model import NNUE
import torch
from torch import nn


def write_feature_transformer(buf: bytearray, nnue: NNUE):
    bias = nnue.ft.bias.data
    bias = bias.mul(127).round().to(torch.int16)
    buf.extend(bias.flatten().numpy().tobytes())

    weight = nnue.ft.weight.data
    weight = weight.mul(127).round().to(torch.int16)
    buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())


def write_layer(buf: bytearray, layer: nn.Linear, output = False):
    if not output:
        s_w, s_b = 64, 127 * 64 
    else:
        s_w, s_b = 64 * 150 / 127, 64 * 150

    bias = layer.bias.data
    bias = bias.mul(s_b).round().to(torch.int32)
    buf.extend(bias.flatten().numpy().tobytes())

    weight = layer.weight.data
    weight = weight.mul(s_w).round().to(torch.int8)
    buf.extend(weight.flatten().numpy().tobytes())


def serialize(source: str, target: str):
    nnue = NNUE()
    nnue.load_state_dict(torch.load(source, 
        map_location=torch.device('cpu')))
    nnue.eval()

    buf = bytearray()
    write_feature_transformer(buf, nnue)
    write_layer(buf, nnue.l1)
    write_layer(buf, nnue.l2)
    write_layer(buf, nnue.l3, output=True)

    with open(target, 'wb') as f:
        f.write(buf)


