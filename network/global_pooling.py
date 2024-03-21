from torch import nn as nn
from model_utils import *

class GlobalPooling(nn.Module):
    def __init__(self, dropout):
        super(GlobalPooling, self).__init__()
        size = [512, 512, 512]
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

    def forward(self, x):

        A_path, h_path = self.path_attention_head(x)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path) #h_path不应该在套个linear嘛
        h_path = self.path_rho(h_path).squeeze()
        return h_path