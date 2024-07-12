import torch.nn as nn
from multihead_attention import MultiHeadAttentionBlock
from feedforward_layer import FeedForwardLayer
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from projection_layer import ProjectionLayer
from positional_encoding import PositionalEncoding
from transformer import Transformer

def build_transformer(seq_len, d_model=512, n_heads=8, d_hidden=2048, n_x=6):
    voc_size = 1000
    d_k = d_model//n_heads
    encoder_blocks = []
    for _ in range(n_x):
        mha = MultiHeadAttentionBlock(d_model, n_heads, d_k)
        ffl = FeedForwardLayer(d_model, d_hidden)
        encoder_block = EncoderBlock((seq_len, d_model), mha, ffl)
        encoder_blocks.append(encoder_block)

    encoder_blocks = nn.ModuleList(encoder_blocks)
    encoder = Encoder(encoder_blocks)

    decoder_blocks = []
    for _ in range(n_x):
        mha = MultiHeadAttentionBlock(d_model, n_heads, d_k)
        ca = MultiHeadAttentionBlock(d_model, n_heads, d_k)
        ffl = FeedForwardLayer(d_model, d_hidden)
        decoder_block = DecoderBlock((seq_len, d_model), mha, ca, ffl)
        decoder_blocks.append(decoder_block)

    decoder_blocks = nn.ModuleList(decoder_blocks)
    decoder = Decoder(decoder_blocks)
    
    src_pos = PositionalEncoding(seq_len, d_model)
    tgt_pos = PositionalEncoding(seq_len, d_model)
    projection_layer = ProjectionLayer(d_model, voc_size)
    transfomer = Transformer(encoder, decoder, projection_layer, src_pos, tgt_pos)
    return transfomer