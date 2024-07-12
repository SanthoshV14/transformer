import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, voc_size:int):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model)
        self.ff2 = nn.Linear(d_model, voc_size)

    def forward(self, x):
        return torch.log_softmax(self.ff2(self.ff1(x)), -1)