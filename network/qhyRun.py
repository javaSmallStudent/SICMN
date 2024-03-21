import math
import random

import numpy as np
import torch

from myDataset import MyDataset
from torch.utils.data import DataLoader
import qhyCode.qhyConfig as mc
import agg_fc
from resnet import resnet18
from resnet import resnet50
from qhy_textEncoder import TextEncoder
import agg_i2gcn
from utils import saveModel, load_model_pth
from utils import returnImportance, cal_score, save_log
from loss_func import drop_consistency_loss, loss_dependence_batch
from model_coattn import *
from global_pooling import GlobalPooling
from FusionLayer import FusionLayer
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from lifelines.utils import concordance_index
from MyModel import qhyModel
from TextTrans import textTrans
from img_text_trans import ImgTextTrans

def main():
    #five-fold
    epoch_max = 150
    c_index = []
    mae = []
    mse = []
    rmse = []
    r2 = []
    mape = []
    for i in range(0, 5):
        c_index_max,mae_min,mse_min,rmse_min,r2_max,mape_min = train(i, epoch_max)
        c_index.append(c_index_max)
        mae.append(mae_min)
        mse.append(mse_min)
        rmse.append(rmse_min)
        r2.append(r2_max)
        mape.append(mape_min)

    c_index_mean = np.mean(c_index)
    mae_mean = np.mean(mae)
    mse_mean = np.mean(mse)
    rmse_mean = np.mean(rmse)
    mape_mean = np.mean(mape)
    r2_mean = np.mean(r2)

    c_index_std = np.std(c_index)
    mae_std = np.std(mae)
    mse_std = np.std(mse)
    rmse_std = np.std(rmse)
    mape_std = np.std(mape)
    r2_std = np.std(r2)

    test_txt = ('%10.4g' * 6) % (c_index_mean, mae_mean, mse_mean, rmse_mean, r2_mean, mape_mean)
    save_log('qhy_mean.txt', test_txt)

    test2_txt = ('%10.4g' * 6) % (c_index_std, mae_std, mse_std, rmse_std, r2_std, mape_std)
    save_log('qhy_std.txt', test2_txt)




def train(k_fold_now, epoch_max):
    print('fold' + str(k_fold_now) + 'start')
    if not os.path.exists(mc.summary_path):
        os.makedirs(mc.summary_path)
    summary_writer_train = SummaryWriter(mc.summary_path + f'/train')
    summary_writer_test = SummaryWriter(mc.summary_path + f'/test')


    train_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='train',
                          ki=0, k=mc.k, k_fold_now=k_fold_now, transform=mc.transforms_train, rand=True)
    test_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='valid',
                         ki=0, k=mc.k, k_fold_now=k_fold_now, transform=mc.transforms_train, rand=True)

    train_loader = DataLoader(train_set, batch_size=mc.patient_batch_size,
                              shuffle=True, num_workers=mc.num_workers)
    test_loader = DataLoader(test_set, batch_size=mc.patient_batch_size,
                             shuffle=True, num_workers=mc.num_workers)

    max_valid_slice_num = train_set.max_valid_slice_num

    #prepare model
    #model = qhyModel(max_valid_slice_num=max_valid_slice_num)
    model = qhyModel(max_valid_slice_num)
    aggregator_pre = agg_fc.Aggregator(survival_len=1, in_dim=mc.in_dim, mean_flag=True)
    imgEncoder = resnet18(pretrained=True)

    #loss and optimizer
    loss = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=mc.lr * mc.lr_factor,
                                           weight_decay=mc.weight_decay,eps=1e-3)
    lr_scheduler_model1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    optimizer_pre = torch.optim.Adam(params=aggregator_pre.parameters(),lr=mc.lr * mc.lr_factor,
                                           weight_decay=mc.weight_decay,eps=1e-3 )
    lr_scheduler_model2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pre, T_max=20, eta_min=1e-6)
    optimizer_encoder = torch.optim.Adam(params=imgEncoder.parameters(),lr=mc.lr * mc.lr_factor,
                                           weight_decay=mc.weight_decay,eps=1e-3 )
    lr_scheduler_model3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=20, eta_min=1e-6)

    c_index_max = 0
    mae_min = 0x3f3f3f3f
    mse_min = 0x3f3f3f3f
    rmse_min = 0x3f3f3f3f
    mape_min = 0x3f3f3f3f
    r2_max = 0

    # --------------------------------------train------------------------------------------------
    for epoch in range(0, epoch_max):
        train_a_epoch(
            model=model,
            aggregator_pre=aggregator_pre,
            imgEncoder=imgEncoder,
            train_loader=train_loader,
            epoch=epoch,
            HuberLoss=loss,
            summary_writer_train=summary_writer_train,
            optimizer=optimizer,
            optimizer_pre=optimizer_pre,
            optimizer_encoder=optimizer_encoder,
            lr_scheduler_model1=lr_scheduler_model1,
            lr_scheduler_model2=lr_scheduler_model2,
            lr_scheduler_model3=lr_scheduler_model3
        )
        c_index, mae, mse, rmse, r2, mape = test(
        test_loader=test_loader,
        model=model,
        aggregator_pre=aggregator_pre,
        imgEncoder=imgEncoder,
        HuberLoss=loss,
        epoch=epoch,
        summary_writer_test=summary_writer_test
        )
        if c_index > c_index_max:
            c_index_max = c_index
        if mae < mae_min:
            mae_min = mae
        if mse < mse_min:
            mse_min = mse
        if rmse < rmse_min:
            rmse_min = rmse
        if r2 > r2_max:
            r2_max = r2
        if mape < mape_min:
            mape_min = mape

    return c_index_max,mae_min,mse_min,rmse_min,r2_max,mape_min


def train_a_epoch(
        model,
        aggregator_pre,
        imgEncoder,
        train_loader,
        epoch,
        HuberLoss,
        summary_writer_train,
        optimizer,
        optimizer_pre,
        optimizer_encoder,
        lr_scheduler_model1,
        lr_scheduler_model2,
        lr_scheduler_model3
                   ):
    model.cuda()
    aggregator_pre.cuda()
    imgEncoder.cuda()

    model.train()
    aggregator_pre.train()
    imgEncoder.train()
    loss_train_history = []


    train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}, Train', colour=mc.color_train)
    for idx, patient_batch in enumerate(train_tqdm):
        image3D = patient_batch['image3D'].to(mc.device)
        text = patient_batch['text'].to(mc.device)
        label_survivals = patient_batch['survivals'].to(mc.device)

        image3D = image3D.squeeze(0)

        data = image3D.to(mc.device)
        data = torch.transpose(data, 0, 1)
        data_shape = data.shape  # B*Z*H*W

        data = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])  # (B*K)*1*H*W

        img = data.to(mc.device)

        #Importance cal
        _, img_features, _ = imgEncoder(x=img)
        img_features = img_features.contiguous().view(data_shape[0], data_shape[1], -1)  # B*Z*512
        img_features = torch.transpose(img_features, 1, 2)

        output_pre = aggregator_pre(img_features)
        loss_pre = HuberLoss(output_pre, label_survivals)

        _, importance_std = returnImportance(
            feature=img_features.clone().detach().data,
            weight_softmax=aggregator_pre.state_dict()['classifier.weight'].data,
            class_idx=[i for i in range(1)])  # B*Z

        importance_std = torch.unsqueeze(
            importance_std,
            dim=1)


        #model
        #output = model(img_features, text, importance_std)
        output = model(img_features, text, importance_std)
        #output = model(text)
        #loss
        loss = HuberLoss(output, label_survivals) + loss_pre
        #loss = HuberLoss(output, label_survivals)

        model.zero_grad()
        imgEncoder.zero_grad()
        aggregator_pre.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer_encoder.step()
        optimizer_pre.step()

        loss_train_history.append(np.array(loss.detach().cpu()))

    loss_train_history = np.array(loss_train_history)
    loss_train_history_mean = loss_train_history.mean(axis=0)
    summary_writer_train.add_scalar('Loss', loss_train_history_mean, epoch + 1)
    lr_scheduler_model1.step()
    lr_scheduler_model2.step()
    lr_scheduler_model3.step()




def test(
        test_loader,
        model,
        imgEncoder,
        aggregator_pre,
        HuberLoss,
        epoch,
        summary_writer_test
):
    model.cuda()
    model.eval()
    with torch.no_grad():
        test_tqdm = tqdm(test_loader, desc=f'Epoch {epoch}, Test', colour=mc.color_test)
        label_survivals_history = []
        predicted_survivals_history = []
        loss_test_history = []

        for idx, patient_batch in enumerate(test_tqdm):
            image3D = patient_batch['image3D'].to(mc.device)
            text = patient_batch['text'].to(mc.device)
            label_survivals = patient_batch['survivals'].to(mc.device)

            image3D = image3D.squeeze(0)

            data = image3D.to(mc.device)
            data = torch.transpose(data, 0, 1)
            data_shape = data.shape  # B*Z*H*W

            data = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])  # (B*K)*1*H*W

            img = data.to(mc.device)
            #
            # # Importance cal
            _, img_features, _ = imgEncoder(x=img)
            img_features = img_features.contiguous().view(data_shape[0], data_shape[1], -1)  # B*Z*512
            img_features = torch.transpose(img_features, 1, 2)
            #
            _, importance_std = returnImportance(
                feature=img_features.clone().detach().data,
                weight_softmax=aggregator_pre.state_dict()['classifier.weight'].data,
                class_idx=[i for i in range(1)])  # B*Z

            importance_std = torch.unsqueeze(
                importance_std,
                dim=1)


            # model
            predicted_survivals = model(img_features, text, importance_std)
            # loss
            loss_survivals = HuberLoss(predicted_survivals, label_survivals)

            test_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

            label_survivals_array = np.array(label_survivals.squeeze(0).detach().cpu())
            predicted_survivals_array = np.array(predicted_survivals.squeeze(0).detach().cpu())
            loss_survivals_array = np.array(loss_survivals.detach().cpu())

            label_survivals_history.append(label_survivals_array)
            predicted_survivals_history.append(predicted_survivals_array)
            loss_test_history.append(loss_survivals_array)

        c_index = np.array([concordance_index(
            np.array(label_survivals_history)[:, i],
            np.array(predicted_survivals_history)[:, i]) for i in range(mc.survivals_len)]).mean(axis=0)
        print('c_index' + str(c_index))
        mae, mse, rmse, r2, mape = cal_score(predicted_survivals_history, label_survivals_history)
        print(r2)
        loss_test_history_mean = np.array(loss_test_history).mean(axis=0)

        summary_writer_test.add_scalar('Loss', loss_test_history_mean, epoch + 1)
        summary_writer_test.add_scalar('C Index', c_index, epoch + 1)

        if epoch >= mc.epoch_start_save_model - 1:

            if c_index > mc.max_test_C_index:
                mc.max_test_C_index = c_index
                saveModel(model=model, epoch=epoch, model_save_path=mc.qhy_model_save_path,
                          loss_test_history_mean=loss_test_history_mean,
                          c_index=c_index, type='co_model')

        return c_index, mae, mse, rmse, r2, mape



if __name__ == '__main__':
    main()



