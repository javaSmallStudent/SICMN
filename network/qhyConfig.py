"""
Some variables

Author: Han
"""
from datetime import datetime

from PIL import Image

from torch import device, cuda
from torchvision import transforms


"""(1) Dataloader"""
size = 332
excel_path = '../dataset/clinical.xlsx'
data_path = '../dataset/mha'
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])
transforms_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])
k_start = 0             # KFold start ki
k = 5                   # KFold k
num_workers = 2         # num_workers of data loader
text_length_dim = 2     # text is of torch.Size((1, 1, 12)) and we take the 2nd num_patches as its length

"""(2) Network"""
is_image = True
is_text = True
is_position = True
is_transformer = True
patient_batch_size = 1
batch_size = 64
lr = 1e-3
weight_decay = 1e-6

date_time = datetime.now().strftime("%Y%m%d%H%M%S")
epoch_description = f'{date_time}_lr={lr}' \
                    f'{"_wo-image" if not is_image else ""}' \
                    f'{"_wo-text" if not is_text else ""}' \
                    f'{"_wo-position" if not is_position else ""}' \
                    f'{"_wo-transformer" if not is_transformer else ""}'
model_resnet_path = 'pretrainedModel/resnet18-5c106cde.pth'
model_path = f'./model/model_{epoch_description}'
model_path_reg = f'./model/model_{epoch_description}/*epoch_*.pth'
test_min_loss_model_path_reg = f'./model/model_{epoch_description}/test_min_loss_epoch_*.pth'
summary_path = f'./log/summary_{epoch_description}'

d_model = 512
in_dim = 512
nhead = 8
num_layers = 6

text_len = 12
survivals_len = 1

epoch_start = 0
epoch_end = 300
epoch_start_save_model = 0
epoch_save_model_interval = 10

min_train_loss = 1e10
min_test_loss = 1e10
max_test_C_index = 0

color_train = '#f14461'
color_test = '#27ce82'


#i2GCN
inter_dim = 512
out_dim = 512
dim_feedforward=1024
sigma = 32
adj_ratio = 0.2
keep_top = 0.8
drop_p = 0.5
gcn_bias = 0



lr_factor = 0.1
#aggregator_pre
ratio_pre = 1.0
ratio_dc = 1000
ratio_HSIC = 100
ratio_rank = 1.0
rank_m = 0.05

#Global Pooling
drop_out = 0.25

qhy_model_save_path = '../model'

threshold = -1e-3
#resume_flag = False

