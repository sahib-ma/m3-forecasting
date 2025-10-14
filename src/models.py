import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_len, output_len, hidden=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_len)
        )
    def forward(self, x):
        return self.net(x)