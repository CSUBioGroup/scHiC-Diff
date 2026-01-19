# cell-cell interaction layer
import torch
import torch.nn as nn



class CellCellInteractionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bbl = nn.Linear(dim, dim)

    def forward(self, x):
        print("进入BBL模块...")
        out = self.bbl(x)
        return out
