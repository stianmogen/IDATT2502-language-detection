import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(inplace=True))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        out = self.softmax(self.linear(inputs))
        return out
