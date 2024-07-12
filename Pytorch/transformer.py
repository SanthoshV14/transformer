import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from projection_layer import ProjectionLayer
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding):
        super().__init__()
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, x, y, tgt_mask):
        y = self.tgt_pos(y)
        return self.decoder(x, y, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, x, src_mask, tgt_mask):
        y = self.encode(x, src_mask)
        z = self.decode(x, y, tgt_mask)
        z = nn.functional.softmax(self.project(z), -1)
        return z