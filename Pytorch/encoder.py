import torch.nn as nn
from multihead_attention import MultiHeadAttentionBlock
from feedforward_layer import FeedForwardLayer
from residual_block import ResidualBlock

class EncoderBlock(nn.Module):
    def __init__(self, normalized_shape, mha: MultiHeadAttentionBlock, ffl: FeedForwardLayer):
        super().__init__()
        self.mha = mha
        self.ffl = ffl
        self.res = nn.ModuleList([ResidualBlock(normalized_shape) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.res[0](x, lambda x: self.mha(x, x, x, src_mask))
        x = self.res[1](x, self.ffl)

        return x

class Encoder(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks

    def forward(self, x, src_mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return x