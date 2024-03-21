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

class ImgTextTrans(nn.Module):
    def __init__(self):
        super(ImgTextTrans, self).__init__()

        self.textEncoder = TextEncoder(mc.text_len, mc.in_dim)
        text_TransformerLayer = MyTransformer.TransformerEncoderLayer(d_model=mc.d_model, nhead=2, dim_feedforward=1024,
                                                   dropout=mc.drop_out,
                                                   activation='relu')


        self.text_co_block = MyTransformer.TransformerEncoder(text_TransformerLayer, num_layers=2)

        self.co_attention = MultiheadAttention(embed_dim=mc.in_dim, num_heads=2)
        #
        #
        img_encoder_layer = nn.TransformerEncoderLayer(d_model=mc.in_dim, nhead=mc.nhead, dim_feedforward=1024,
                                                       dropout=mc.drop_out,
                                                       activation='relu')

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=mc.in_dim, nhead=mc.nhead, dim_feedforward=1024,
                                                        dropout=mc.drop_out,
                                                        activation='relu')

        self.img_transformer = nn.TransformerEncoder(img_encoder_layer, num_layers=4)

        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=3)


        self.fc = nn.Linear(1024, 1)



    def forward(self, img_features, text):
        img_features = img_features

        text_features = self.textEncoder(text)


        text_input = text_features.unsqueeze(0)

        img_input = img_features.contiguous().view(img_features.shape[2], img_features.shape[0],
                                                   img_features.shape[1])



        img_co = self.co_attention(text_input, img_input, img_input)
        #text_co = self.co_attention2(q, text_input, text_input)
        #

        img_trans = self.img_transformer(img_co)
        text_trans = self.text_transformer(text_input)

        # if self.max_valid_slice_num > img_trans.shape[0]:
        #     zeros = torch.zeros([self.max_valid_slice_num - img_trans.shape[0], 1, mc.d_model],
        #                         dtype=torch.float32).to(mc.device)
        #     x_list = []
        #     x_list.append(img_trans)
        #     x_list.append(zeros)
        #     img_trans = torch.cat(x_list, 0)

        x_list = []
        x_list.append(text_trans)
        x_list.append(img_trans)
        fusion = torch.cat(x_list, -1).squeeze(0)


        #classfier


        output = self.fc(fusion)
        output = torch.sigmoid(output)

        return output

if __name__ == '__main__':
    text = torch.randn(1, 12)
    img = torch.randn(1, 512, 75)
    model = ImgTextTrans()
    output = model(img, text)
    print(output)

