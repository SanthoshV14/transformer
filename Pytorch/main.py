import torch
import torch.nn as nn
from model import build_transformer
from tokenizer import Tokenizer

if __name__ == '__main__':
    
    # Build Model
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_x = 6
    d_hidden = 4*d_model
    transformer = build_transformer(seq_len, d_model, n_heads, d_hidden, n_x)
    
    # Prepare Sample Data
    padd_token = '<PAD>'
    vocab_size = 1000
    tokenizer = Tokenizer(seq_len, padd_token)
    tokens = tokenizer.tokenize(['hi how are you', 'hello world hi I am santhosh'])
    ids = tokenizer.unique_ids(tokens)
    embedding = nn.Embedding(vocab_size, d_model)
    embeddings = embedding(ids)
    print(tokens, ids.shape, embeddings.shape)

    # Test model
    src_mask = (ids != 0).unsqueeze(1).unsqueeze(-1)
    tgt_mask = torch.ones((seq_len, seq_len)).tril(diagonal=0)
    tgt_mask = (src_mask * tgt_mask) == 0
    res = transformer(embeddings, src_mask, tgt_mask)
    print(res.shape, torch.argmax(res, dim=-1))    