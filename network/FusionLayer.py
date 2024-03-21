from torch import nn as nn
import torch
import qhyCode.qhyConfig as mc
class FusionLayer(nn.Module):
    def __init__(self, max_valid_slice_num):
        super(FusionLayer, self).__init__()
        size = mc.d_model
        self.globalSummary = nn.Sequential(*[nn.Linear(max_valid_slice_num, 1)], nn.LeakyReLU()) # [B,K,512] > [B,1,512]
        self.globalSummary2 = nn.Sequential(*[nn.Linear(max_valid_slice_num, 1)], nn.LeakyReLU()) # [B,K,512] > [B,1,512]
        self.mm = nn.Sequential(*[nn.Linear(size * 2, size), nn.LeakyReLU()])#[B, 1, 1024] > [B, 1, 512]
        self.classifier = nn.Linear(size, 1) #[B,1,512] > [B, 1,1]

    def forward(self, a, b):
        b = b.transpose(0, 1)
        b = self.globalSummary(b)
        b = b.transpose(0, 1)

        a = a.transpose(0, 1)
        a = self.globalSummary2(a)
        a = a.transpose(0, 1)


        x = self.mm(torch.cat([a, b], axis=1))
        x = self.classifier(x)
        return x
