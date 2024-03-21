"""
TextEncoder class in the model.

Author: Han
"""
import torch
import torch.nn as nn

import qhyConfig as mc


class TextEncoder(nn.Module):
    """
    Encode text.
    """
    def __init__(self, in_channel, out_channel):
        super(TextEncoder, self).__init__()
        #self.fc = nn.Linear(mc.text_len, mc.text_len)
        self.fc1 = nn.Linear(in_channel, out_channel)
        # self.fc2 = nn.Linear(256, out_channel)
        # self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        return x


if __name__ == '__main__':
    # print(TextEncoder())
    text_encoder = TextEncoder(12, 512).to(mc.device)
    text = torch.rand(1, mc.text_len, dtype=torch.float32).to(mc.device)
    text_feature = text_encoder(text)
    print(text_feature.shape)
    print(text_feature)
