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
from itertools import combinations

from Dataloader import PsdDataset, load_data, make_label, data_reshape
from Loss import ova_ent, ova_loss
from Model import MLP

# ----- ----- Device Setting ----- -----
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device} device.\n')


# ----- ----- ----- Dataset ----- ----- -----
channels = np.array(['M1', 'Pu', 'Gpe', 'VAL', 'VLL', 'Gpi', 'STN', 'SNR'])

dataset_path = {
    'PD': r'Data/PD',
    'LID': r'Data/LID',
    'Nor': r'Data/Nor'
}

train_data_raw = []
train_label = []

test_data_raw = []
test_label = []

for key, path in dataset_path.items():
    files = os.listdir(path)
    for file in files:
        data = load_data(os.path.join(path, file))

        length = data.shape[0]

        train_data_raw.append(data[:int(length * 0.7), :, :])
        train_label.extend(make_label(data[:int(length * 0.7), :, :], key))

        test_data_raw.append(data[int(length * 0.7):, :, :])
        test_label.extend(make_label(data[int(length * 0.7):, :, :], key))

sites = [0, 4, 6, 7]
new_sites = []
for s in range(len(sites)):
    for combination in combinations(sites, s + 1):
        new_sites.append(list(combination))

loops = 10
epochs = 100

for loop in range(loops):

    if os.path.exists(str(loop)) is False:
        os.mkdir(str(loop))

    print("\n" + "-" * 20 + f" Loop {loop + 1} " + "-" * 20 + "\n")

    num_index = 1

    df = {
        "Name": [],
        "Site Number": [],
        "Close Accuracy": [],
        "Open Accuracy": []
    }

    print("Order    "
          "Time(s)    "
          "Accuracy(close)    "
          "Accuracy(open)    "
          "Site")

    for site in new_sites:
        start_time = time.time()

        # Name
        name = ""

        for i in channels[site]:
            name = name + i + '_'

        name = name[:-1]

        df["Name"].append(name)
        df["Site Number"].append(len(site))

        train_data, _ = data_reshape(np.concatenate(train_data_raw, axis=0), site)
        test_data, feature_shape = data_reshape(np.concatenate(test_data_raw, axis=0), site)

        train_dataset = PsdDataset(train_data, train_label)
        test_dataset = PsdDataset(test_data, test_label)

        # Num Worker May Be the Number of CPU, Now set 0
        num_worker = 0

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=num_worker
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=num_worker
        )

        model = MLP(input_dim=feature_shape, output_dim=int(2 * feature_shape), classes=2)
        model = model.to(device)

        optimizer = Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train_loss = []
        valid_loss = []
        best_loss = None

        accuracy_open = []
        accuracy_close = []
        accuracy_wo_close = []
        accuracy_wo_open = []
        best_accuracy_open = None
        best_accuracy_close = None
        best_wo_accuracy_close = None
        best_wo_accuracy_open = None

        for epoch in range(epochs):

            loss_sum = 0
            loss_index = 0

            model.train()
            for X, y in train_dataloader:
                X = X.to(device)
                y = y.to(device)

                pred, prob = model(X)

                optimizer.zero_grad()

                ood_index = np.where(y.cpu() == 2)[0]
                in_index = np.where(y.cpu() < 2)[0]

                in_pred = pred[in_index]
                in_y = y[in_index]

                loss_pred = loss_fn(in_pred, in_y)
                loss_ent = ova_ent(prob)
                loss_ova = ova_loss(prob, y, ood_index)

                loss = loss_ova + 0.1 * loss_ent + loss_pred

                loss_sum = loss.cpu().data.numpy() + loss_sum
                loss_index = loss_index + 1

                loss.backward()
                optimizer.step()

            train_loss.append(loss_sum / loss_index)

            loss_sum = 0
            loss_index = 0

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

                    loss_pred = loss_fn(in_pred, in_y)
                    loss_ent = ova_ent(prob)
                    loss_ova = ova_loss(prob, y, ood_index)

                    loss = loss_ova + 0.1 * loss_ent + loss_pred

                    loss_sum = loss.cpu().data.numpy() + loss_sum
                    loss_index = loss_index + 1

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
                    count_all = count_all + len(y)

                    correct_close = correct_close + (pred_close == y).sum().item()
                    correct_open = correct_open + (pred_open == y).sum().item()
                    correct_wo_close = correct_wo_close + (pred_close[in_index] == in_y).sum().item()
                    correct_wo_open = correct_wo_open + (pred_open[in_index] == in_y).sum().item()

                valid_loss.append(loss_sum / loss_index)
                accuracy_close.append(correct_close / count_all)
                accuracy_open.append(correct_open / count_all)
                accuracy_wo_close.append(correct_wo_close / count_without_2)
                accuracy_wo_open.append(correct_wo_open / count_without_2)

                if best_loss is None:
                    best_loss = valid_loss[-1]
                    best_accuracy_close = accuracy_close[-1]
                    best_accuracy_open = accuracy_open[-1]
                    best_wo_accuracy_close = accuracy_wo_close[-1]
                    best_wo_accuracy_open = accuracy_wo_open[-1]
                    model_param = model.state_dict()
                else:
                    if best_loss >= valid_loss[-1]:
                        best_loss = valid_loss[-1]
                        best_accuracy_close = accuracy_close[-1]
                        best_accuracy_open = accuracy_open[-1]
                        best_wo_accuracy_close = accuracy_wo_close[-1]
                        best_wo_accuracy_open = accuracy_wo_open[-1]
                        model_param = model.state_dict()

        stop_time = time.time()
        current_time = stop_time - start_time

        torch.save(model_param, os.path.join(str(loop), name + '.pkl'))

        df["Close Accuracy"].append(best_wo_accuracy_close)
        df["Open Accuracy"].append(best_wo_accuracy_open)

        print(f" {num_index}" + " " * (8 - len(str(num_index))) +
              f" {current_time:.2f} "
              f"    {best_wo_accuracy_close:.4f}         "
              f"    {best_wo_accuracy_open:.4f}        "
              f"{name}")

        num_index = num_index + 1

    df = pd.DataFrame(df)
    df.to_excel(os.path.join(str(loop), 'Result.xlsx'))
