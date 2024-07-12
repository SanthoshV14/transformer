import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, d_k):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_model = self.h * self.d_k
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask):
        k = k.transpose(-2, -1)
        attention_scores = torch.matmul(q, k)/math.sqrt(q.shape[-1])
        if mask is not None:
            attention_scores = torch.masked_fill(attention_scores, mask, -1e-9)
        attention_scores = nn.functional.softmax(attention_scores, -1)
        attention = torch.matmul(attention_scores, v)
        return attention

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = torch.reshape(q, (q.shape[0], q.shape[1], self.h, self.d_k)).transpose(1, 2)
        k = torch.reshape(k, (k.shape[0], k.shape[1], self.h, self.d_k)).transpose(1, 2)
        v = torch.reshape(v, (v.shape[0], v.shape[1], self.h, self.d_k)).transpose(1, 2)
        x = MultiHeadAttentionBlock.attention(q, k, v, mask)

        x = x.transpose(1, 2)
        x = x.reshape((x.shape[0], x.shape[1], self.d_model))

        return self.w_o(x)
    