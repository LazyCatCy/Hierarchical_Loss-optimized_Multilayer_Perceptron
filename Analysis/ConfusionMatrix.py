# Pytorch One Vs All 混淆矩阵
import os
import copy
import numpy as np
import time
import torch

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from Dataloader import PsdDataset, load_data, make_label, data_reshape
from Model import MLP
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12

# 保存SVG时使用<text>元素来表示文本
mpl.rcParams['svg.fonttype'] = 'none'
# 保证同一Matplotlib图形在不同计算机上生成的SVG文件HASH值相同，从而便于比较和合并文件
mpl.rcParams['svg.hashsalt'] = 'hello'

dataset_path = {
    'PD': r'C:\Users\LazyCat\Desktop\Article\2 Figure\1 Rats\模型画图\Data\PD_power',
    'LID': r'C:\Users\LazyCat\Desktop\Article\2 Figure\1 Rats\模型画图\Data\LID_power',
    'Nor': r'C:\Users\LazyCat\Desktop\Article\2 Figure\1 Rats\模型画图\Data\SD'
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

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device} device.\n')

model = MLP(input_dim=feature_shape, output_dim=int(2 * feature_shape), classes=2)
model.load_state_dict(torch.load(r"C:\Users\LazyCat\Desktop\Article\2 Figure\1 Rats\模型画图\2K_Result\0\M1_DS_Gpe_Tha_Gpi_STN_SNR_PPN.pkl", map_location=torch.device('cpu')))
model = model.to(device)

count_without_2 = 0
count_all = 0

correct_close = 0
correct_open = 0
correct_wo_close = 0
correct_wo_open = 0

matrix = np.zeros((2, 3))

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

        for i in range(len(in_y)):
            matrix[in_y[i], pred_open[in_index][i]] += 1

# 混淆矩阵
matrix[0, :] = matrix[0, :] / np.sum(matrix[0, :])
matrix[1, :] = matrix[1, :] / np.sum(matrix[1, :])

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(matrix, cmap="GnBu", aspect="equal", origin="upper")

ax.set_xticks(np.arange(3), labels=["PD", "LID", "ATS"])
ax.set_yticks(np.arange(2), labels=["PD", "LID"])

cbar = ax.figure.colorbar(im, ax=ax)

for i in range(2):
    for j in range(3):
        text = f"{matrix[i, j] * 100:.2f} %"

        # Color
        color = None
        if matrix[i, j] >= 0.7:
            color = "white"
        else:
            color = "black"

        ax.text(j, i, text, ha="center", va="center", color=color)

ax.set_xlabel('Predict')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

cbar.set_label('Accuracy')
fig.tight_layout()
fig.patch.set_alpha(0.)
fig.savefig('Confusion.svg')
