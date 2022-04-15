import torch
from torch import nn

NUM_FEATURES = 40960

def crelu(x):
    return torch.clamp(x, 0.0, 1.0)

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, 4)
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, 1)

        self.weight_clipping = [
            {'params' : [self.ft.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
            {'params' : [self.l3.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
        ]

    def clip_weights(self):
        for group in self.weight_clipping:
            for p in group['params']:
                p_data_fp32 = p.data
                min_weight = group['min_weight']
                max_weight = group['max_weight']
                p_data_fp32.clamp_(min_weight, max_weight)
                p.data.copy_(p_data_fp32)

    def forward(self, wfts, bfts, stm):
        w = self.ft(wfts)
        b = self.ft(bfts)

        accumulator = stm * torch.cat([w, b], dim=1)
        accumulator += (1 - stm) * torch.cat([b, w], dim=1)

        x = crelu(accumulator)
        x = crelu(self.l1(x))
        x = crelu(self.l2(x))
        x = self.l3(x)

        return x


