import torch
import torch.nn as nn

class Tokenizer(nn.Module):
    def __init__(self, seq_len, padd_token):
        super().__init__()
        self.padd_token = padd_token
        self.seq_len = seq_len
        self.embedding = {
            padd_token: 0
        }
        self.counter = 0

    def tokenize(self, sentences):
        tokens = []
        for sentence in sentences:
            sentence = list(set(sentence.lower().split()))
            tokens.append(sentence)

        return tokens
    
    def unique_ids(self, tokens):
        embeddings = []
        for token_arr in tokens:
            if len(token_arr) < self.seq_len:
                token_arr = token_arr + [self.padd_token]*(self.seq_len-len(token_arr))
            else:
                token_arr = token_arr[:self.seq_len]

            for i, token in enumerate(token_arr):
                if token not in self.embedding:
                    self.counter = self.counter+1
                    self.embedding[token] = self.counter
                token_arr[i] = self.embedding[token]
            embeddings.append(token_arr)
        
        return torch.tensor(embeddings, dtype=torch.int)
    
    def token_to_id(self, token: str):
        return self.embedding[token]
    
    def id_to_token(self, id: int):
        id_index = list(self.embedding.values()).index(id)
        token = list(self.embedding.keys())[id_index]
        return token