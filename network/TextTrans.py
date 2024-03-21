import torch.nn as nn
from myDataset import MyDataset
from torch.utils.data import DataLoader
import qhyCode.qhyConfig as mc
import agg_fc
from resnet import resnet18
from resnet import resnet50
from qhy_textEncoder import TextEncoder
import agg_i2gcn
from utils import saveModel, load_model_pth
from utils import returnImportance
from loss_func import drop_consistency_loss, loss_dependence_batch
from model_coattn import *
from global_pooling import GlobalPooling
from FusionLayer import FusionLayer
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from lifelines.utils import concordance_index
from resNetEncoder import ResNetEncoder
from einops import repeat
import MyTransformer

class textTrans(nn.Module):
    def __init__(self):
        super(textTrans, self).__init__()

        self.textEncoder = TextEncoder(mc.text_len, mc.in_dim)


        #
        #


        text_encoder_layer = nn.TransformerEncoderLayer(d_model=mc.in_dim, nhead=2, dim_feedforward=1024,
                                                        dropout=mc.drop_out,
                                                        activation='relu')


        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=4)


        self.fc = nn.Linear(512, 1)


    def forward(self, text):

        text_features = self.textEncoder(text)


        text_input = text_features.unsqueeze(0)


        text_trans = self.text_transformer(text_input)


        output = self.fc(text_trans)
        output = torch.sigmoid(output).squeeze(0)

        return output


if __name__ == '__main__':
    text = torch.randn(1, 12)
    model = textTrans()
    output = model(text)
    print(output)

