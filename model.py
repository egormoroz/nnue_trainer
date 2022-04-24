import torch
from torch import nn

from psqts import halfka_psqts

NUM_FEATURES = 40960

def crelu(x):
    return torch.clamp(x, 0.0, 1.0)

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, 257)
        self.l1 = nn.Linear(512, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

        self.init_psqt()

        self.weight_clipping = [
            # {'params' : [self.ft.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l3.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
        ]

    def init_psqt(self):
        input_weight = self.ft.weight.data
        input_bias = self.ft.bias.data
        with torch.no_grad():
            psqts = halfka_psqts()
            input_weight[-1, :] = torch.FloatTensor(psqts) / 150.0
            input_bias[:-1] = 0.0
        self.ft.weight.data = input_weight
        self.ft.bias.data = input_bias

    def clip_weights(self):
        # self.ft.weight.data[:-1, :].clamp_(-127/64, 127/64)
        for group in self.weight_clipping:
            for p in group['params']:
                p_data_fp32 = p.data
                min_weight = group['min_weight']
                max_weight = group['max_weight']
                p_data_fp32.clamp_(min_weight, max_weight)
                p.data.copy_(p_data_fp32)

    def forward(self, wfts, bfts, stm):
        wp = self.ft(wfts)
        bp = self.ft(bfts)

        w, wpsqt = torch.split(wp, wp.shape[1]-1, dim=1)
        b, bpsqt = torch.split(bp, bp.shape[1]-1, dim=1)

        accumulator = stm * torch.cat([w, b], dim=1)
        accumulator += (1 - stm) * torch.cat([b, w], dim=1)

        x = crelu(accumulator)
        x = crelu(self.l1(x))
        x = crelu(self.l2(x))
        x = self.l3(x)

        return x + (wpsqt + bpsqt) * (stm - 0.5)


