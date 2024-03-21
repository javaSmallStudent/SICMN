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

class qhyModel(nn.Module):
    def __init__(self, max_valid_slice_num, is_position=True):
        super(qhyModel, self).__init__()

        self.textEncoder = TextEncoder(mc.text_len, mc.in_dim)
        text_TransformerLayer = MyTransformer.TransformerEncoderLayer(d_model=mc.d_model, nhead=2, dim_feedforward=1024,
                                                   dropout=mc.drop_out,
                                                   activation='relu')


        self.text_co_block = MyTransformer.TransformerEncoder(text_TransformerLayer, num_layers=2)

        self.co_attention = MultiheadAttention(embed_dim=mc.in_dim, num_heads=2)
        self.co_attention2 = MultiheadAttention(embed_dim=mc.in_dim, num_heads=2)
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


        self.fusion_classifier = FusionLayer(max_valid_slice_num)
        self.max_valid_slice_num = max_valid_slice_num
        self.is_position = is_position
        if is_position:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 1, 512))


    def forward(self, img_features, text, importance):
        img_features = img_features

        text_features = self.textEncoder(text)


        text_input = text_features.unsqueeze(0)
        # text:q
        q = importance * img_features
        q = q.contiguous().view(q.shape[2], q.shape[0],
                                                    q.shape[1])
        #importance = importance.transpose(0, 2)

        img_input = img_features.contiguous().view(img_features.shape[2], img_features.shape[0],
                                                   img_features.shape[1])

        #img_co, _ = self.img_co_block(img_input, text_input, importance)
        # if self.is_position:
        #     pos_embedding = repeat(self.pos_embedding, '() b z -> k b z', k=img_input.shape[0])
        #     img_input = img_input + pos_embedding

        #img_co = self.co_attention(text_input, img_input, img_input)

        text_co,_ = self.text_co_block(text_input, img_input , importance)
        #

        img_trans = self.img_transformer(img_input)
        text_trans = self.text_transformer(text_co)

        if self.max_valid_slice_num > img_trans.shape[0]:
            zeros = torch.zeros([self.max_valid_slice_num - img_trans.shape[0], 1, mc.d_model],
                                dtype=torch.float32).to(mc.device)
            x_list = []
            x_list.append(img_trans)
            x_list.append(zeros)
            img_trans = torch.cat(x_list, 0)

        if self.max_valid_slice_num > text_trans.shape[0]:
            zeros = torch.zeros([self.max_valid_slice_num - text_trans.shape[0], 1, mc.d_model],
                                dtype=torch.float32).to(mc.device)
            x_list = []
            x_list.append(text_trans)
            x_list.append(zeros)
            text_trans = torch.cat(x_list, 0)


        #classfier


        output = self.fusion_classifier(img_trans.squeeze(), text_trans.squeeze())
        output = torch.sigmoid(output)

        return output



