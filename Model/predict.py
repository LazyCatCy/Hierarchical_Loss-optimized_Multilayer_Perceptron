import os
import copy
import numpy as np
import time
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from Dataloader import PsdDataset, load_data, make_label, data_reshape
from Model import MLP
import matplotlib.pyplot as plt


dataset_path = {
    # 'PD': r'D:\虚空城堡\论文图片202505\模型画图\Data\PD_power',
    'LID': r'D:\虚空城堡\论文图片202505\模型画图\Data\LID_power',
    'Nor': r'D:\虚空城堡\论文图片202505\模型画图\Data\SD'
}

test_data = []
test_label = []

for key, path in dataset_path.items():
    files = os.listdir(path)
    for file in files:
        data = load_data(os.path.join(path, file))

        length = data.shape[0]

        test_data.append(data[int(length * 0.7):, :, :])
        test_label.extend(make_label(data[int(length * 0.7):, :, :], key))

site = [0, 1, 2, 3, 4, 5, 6, 7]

test_data, feature_shape = data_reshape(np.concatenate(test_data, axis=0), site)

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device} device.\n')

model = MLP(input_dim=feature_shape, output_dim=int(2 * feature_shape), classes=2)
model.load_state_dict(torch.load(r"D:\虚空城堡\论文图片202505\模型画图\2K_Result\0\M1_DS_Gpe_Tha_Gpi_STN_SNR_PPN.pkl", map_location=torch.device('cpu')))
model = model.to(device)

test_dataset = PsdDataset(test_data, test_label)

# Num Worker May Be the Number of CPU, Now set 0
num_worker = 0

test_dataloader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=True,
    drop_last=False,
    num_workers=num_worker
)

count_without_2 = 0
count_all = 0

correct_close = 0
correct_open = 0
correct_wo_close = 0
correct_wo_open = 0

model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)

        pred, prob = model(X)

        ood_index = np.where(y.cpu() == 2)[0]
        in_index = np.where(y.cpu() < 2)[0]

        in_pred = pred[in_index]
        in_y = y[in_index]

        output = F.softmax(pred, dim=1)
        output_open = F.softmax(prob.view(prob.size(0), 2, -1), dim=1)

        tmp_range = torch.arange(0, output_open.size(0) - 0.9).long()
        tmp_range = tmp_range.to(device)

        # 模型预测
        pred_close = output.data.max(1)[1]

        unknown_score = output_open[tmp_range, 0, pred_close]
        known_score = output.max(1)[1]

        targets_unknown = y >= int(output.size(1))
        y[targets_unknown] = int(output.size(1))

        targets_known = y < int(output.size(1))
        known_pred = output[targets_known]
        targets_known = y[targets_known]

        # 寻找 ood 数据
        ind_unk = unknown_score > 0.5
        pred_open = copy.deepcopy(pred_close)
        pred_open[ind_unk] = int(output.size(1))

        count_without_2 = count_without_2 + len(in_y)

        correct_wo_open = correct_wo_open + (pred_open[in_index] == in_y).sum().item()

accuracy_wo_open = correct_wo_open / count_without_2
print(accuracy_wo_open)
