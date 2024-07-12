import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        pos = torch.arange(0, seq_len).reshape((seq_len, 1))
        index = torch.arange(0, d_model).reshape((1, d_model))
        denominator = torch.pow(10000, (2*index)/d_model)
        pe = torch.zeros((seq_len, d_model))
        pe[:,0::2] = torch.sin(pos/denominator[:, 0::2])
        pe[:,1::2] = torch.cos(pos/denominator[:, 1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe.requires_grad_(False)