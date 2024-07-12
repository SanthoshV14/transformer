import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self, x, sublayer):
        x = self.layernorm(x + sublayer(x))
        return x