import torch 
import torch.nn as nn
import torch.nn.functional as F 
from qhyCode import gclayer as gclayer
import numpy as np 
import copy
from torch.nn import LayerNorm

def SGA(
    batch_importance, 
    keep_top=0.2, 
    drop_p=0.5,
    is_cuda=False):
    """
    keep_top: p_key
    drop_p: p_retain
    """
    B, _, K = batch_importance.shape # [B, 1, K]
    num_top = int(max(keep_top*K+1, 1)) # key nodes
    num_rest = int(K - num_top) 
    num_drop = int(max(num_rest * drop_p + 0.5, 1)) # drop nodes
    num_kept = int(max(num_rest - num_drop, 1)) # retain nodes

    top_mask_list = [] 
    kept_mask_list = []
    drop_mask_list = [] 
    mask_list = []
    for b in range(B):
        top_mask = torch.zeros(K) 
        kept_mask = torch.zeros(K) 
        drop_mask = torch.zeros(K) 
        # 
        _, indice_sorted = torch.sort(
            batch_importance[b].clone().detach().cpu().view(-1), 
            dim=-1, 
            descending=True) 
        top_mask[indice_sorted[:num_top]] = 1 
        top_mask_list.append(top_mask)
        #
        random_mask = torch.rand(K)
        random_mask1, random_mask2 = random_mask.clone().detach(), random_mask.clone().detach()
        random_mask1[indice_sorted[:num_top]] = -10.0 
        random_mask2[indice_sorted[:num_top]] = 10 
        #
        _, kept_idx_sorted = torch.sort(
            random_mask1, 
            dim=-1, 
            descending=True)
        kept_mask[kept_idx_sorted[:num_kept]] = 1 #
        kept_mask_list.append(kept_mask)
        #
        _, drop_idx_sorted = torch.sort(
            random_mask2, 
            dim=-1, 
            descending=False)
        drop_mask[drop_idx_sorted[:num_drop]] = 1 
        drop_mask_list.append(drop_mask)

    if is_cuda:
        batch_top_mask = torch.stack(top_mask_list, dim=0).unsqueeze(dim=1).cuda() # [B,1,K]
        batch_kept_mask = torch.stack(kept_mask_list, dim=0).unsqueeze(dim=1).cuda()
        batch_drop_mask = torch.stack(drop_mask_list, dim=0).unsqueeze(dim=1).cuda()
    else:
        batch_top_mask = torch.stack(top_mask_list, dim=0).unsqueeze(dim=1) # [B,1,K]
        batch_kept_mask = torch.stack(kept_mask_list, dim=0).unsqueeze(dim=1)
        batch_drop_mask = torch.stack(drop_mask_list, dim=0).unsqueeze(dim=1)

    batch_mask = batch_top_mask + batch_kept_mask
    drop_scale = 1.0 - float(num_drop)/K 

    return batch_mask, drop_scale, (batch_top_mask, batch_kept_mask, batch_drop_mask)


class I2GCN(nn.Module):
    def __init__(self, 
        class_num=1,
        in_dim=512, 
        inter_dim=128, 
        out_dim=64, 
        node_num=64, #保留的batch的size
        sigma=32,
        layer_nums=2,
        adj_ratio=0.2,
        self_loop_flag=True,
        keep_top=0.2, 
        drop_p=0.5,
        gc_bias=False
        ):
        super(I2GCN, self).__init__()
        # self.gc_combine1 = gclayer.I2GCLayer(
        #     in_channel=in_dim,
        #     out_channel=inter_dim,
        #     bias=gc_bias
        # )
        # self.gc_combine2 = gclayer.I2GCLayer(
        #     in_channel=inter_dim,
        #     out_channel=out_dim,
        #     bias=gc_bias
        # )
        # self.gc_combine3 = gclayer.I2GCLayer(
        #     in_channel=inter_dim,
        #     out_channel=out_dim,
        #     bias=gc_bias
        # )
        self.I2GCNBlocks = [gclayer.I2GCLayer(
            in_channel=inter_dim,
            out_channel=out_dim,
            bias=gc_bias
        ).cuda()]
        if layer_nums > 1:
            for i in range(layer_nums - 1):
                self.I2GCNBlocks += [gclayer.I2GCLayer(
                in_channel=inter_dim,
                out_channel=out_dim,
                bias=gc_bias
                ).cuda()]

        self.fc = nn.Linear(out_dim, class_num, bias=False)
        #self.identity = nn.Linear(in_dim, out_dim, bias=True)
        self.identity = nn.Linear(out_dim, out_dim, bias=True)
        self.updatex = nn.Sequential(nn.Linear(inter_dim, out_dim), nn.LeakyReLU(inplace=True))

        self.sigma = sigma
        self.adj_ratio = adj_ratio
        self.self_loop_flag = self_loop_flag
        self.keep_top = keep_top
        self.drop_p = drop_p
        self.norm = nn.BatchNorm1d(512)
        
    def forward(self, x, importances_batch, batch_size):
        self_loop = torch.eye(batch_size)
        # x: [B, C, K]
        # importances_batch: [B, 1, K]


        for I2GCNLayer in self.I2GCNBlocks:
            # x = x * importances_batch
            # x = self.updatex(torch.permute(x, (0, 2, 1)))
            # x = torch.permute(x, (0, 2, 1))
            h = x.clone().cuda()
            adj_prior = gclayer.build_adjacent_PriorImportance(
                x,
                importances_batch=importances_batch,
                sigma=self.sigma,
                self_loop=self_loop,
                self_loop_flag=self.self_loop_flag
            )
            adj_feature = gclayer.build_adjacent_Feature(
                x,
                adj_ratio=self.adj_ratio,
                self_loop=self_loop, #self.loop对角阵
                self_loop_flag=self.self_loop_flag
            )
            x = I2GCNLayer(x, adjacency_matrix_list=[adj_prior, adj_feature])


        return x + h * importances_batch

        # adjacent matrices for 1st I2GCLayer
        # adj_prior_1 = gclayer.build_adjacent_PriorImportance(
        #     x,
        #     importances_batch=importances_batch,
        #     sigma=self.sigma,
        #     self_loop=self_loop,
        #     self_loop_flag=self.self_loop_flag
        # )
        # adj_feature_1 = gclayer.build_adjacent_Feature(
        #     x,
        #     adj_ratio=self.adj_ratio,
        #     self_loop=self_loop, #self.loop对角阵
        #     self_loop_flag=self.self_loop_flag
        # )
        # # 1st I2GCLayer
        # emb1 = self.gc_combine1(x, adjacency_matrix_list=[adj_prior_1, adj_feature_1])
        #
        # # adjacent matrices for 2nd I2GCLayer
        # adj_prior_2 = gclayer.build_adjacent_PriorImportance(
        #     emb1,
        #     importances_batch=importances_batch,
        #     sigma=self.sigma,
        #     self_loop=self_loop,
        #     self_loop_flag=self.self_loop_flag
        # )
        # adj_feature_2 = gclayer.build_adjacent_Feature(
        #     emb1,
        #     adj_ratio=self.adj_ratio,
        #     self_loop=self_loop,
        #     self_loop_flag=self.self_loop_flag
        # )
        
        # 2nd I2GCLayer
        # emb2 = self.gc_combine2(emb1, adjacency_matrix_list=[adj_prior_2, adj_feature_2])


        # 去掉SGA
        # if self.gc_combine1.training:
        #     emb1_kept = emb1*torch.clamp(
        #         (batch_top_mask+batch_kept_mask),
        #         min=0,
        #         max=1)/drop_scale #
        #     bag_kept = torch.sum(
        #         emb1_kept*importances_batch,
        #         dim=-1,
        #         keepdim=False) #
        #     # [B, C]
        #     emb2_kept = self.gc_combine2(emb1_kept, adjacency_matrix_list=[adj_prior_2, adj_feature_2])
        #
        #     emb1_drop = emb1*torch.clamp(
        #         (batch_top_mask+batch_drop_mask),
        #         min=0,
        #         max=1)/drop_scale
        #     bag_drop = torch.sum(
        #         emb1_drop*importances_batch,
        #         dim=-1,
        #         keepdim=False) #
        #     #
        #     emb2 = emb2_kept*batch_mask
        # else:
        #     emb2 = self.gc_combine2(emb1, adjacency_matrix_list=[adj_prior_2, adj_feature_2])
        #

        # emb2 = torch.sigmoid(emb2)
        # latex = torch.sigmoid(x * importances_batch)
        # emb2 = emb2 + latex


        # bag_features = torch.sum(
        #     emb2*importances_batch,
        #     dim=-1,
        #     keepdim=False
        #     )
        # [B, C] #
        # output = self.fc(
        #     bag_features + F.relu(
        #         self.identity(torch.sum(x*importances_batch, dim=-1, keepdim=False))
        #         )
        # )

        #return emb1, emb2, (bag_kept, bag_drop)
        #return emb1, emb2

        #return emb1, emb3 + x * importances_batch

        # if self.gc_combine1.training:
        #     return emb1, emb2, (bag_kept, bag_drop)
        # else:
        #     return emb1, emb2
    
