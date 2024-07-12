import torch.nn as nn
from multihead_attention import MultiHeadAttentionBlock
from feedforward_layer import FeedForwardLayer
from residual_block import ResidualBlock

class DecoderBlock(nn.Module):
    def __init__(self, normalized_shape, mha: MultiHeadAttentionBlock, ca: MultiHeadAttentionBlock, ffl: FeedForwardLayer):
        super().__init__()
        self.mha = mha
        self.ca = ca
        self.ffl = ffl
        self.res = nn.ModuleList([ResidualBlock(normalized_shape) for _ in range(3)])

    def forward(self, x, y, tgt_mask):
        x = self.res[0](x, lambda x: self.mha(x, x, x, tgt_mask))
        x = self.res[1](x, lambda x: self.ca(y, y, x))
        x = self.res[2](x, self.ffl)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, decoder_blocks: nn.ModuleList):
        super().__init__()
        self.decoder_blocks = decoder_blocks

    def forward(self, x, y, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, y, tgt_mask)
        
        return x