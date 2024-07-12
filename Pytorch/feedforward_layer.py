import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model:int, d_hidden:int):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_hidden)
        self.ff2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ff1(x))
        return self.ff2(x)