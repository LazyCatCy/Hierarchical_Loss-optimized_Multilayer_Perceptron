import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14

# 保存SVG时使用<text>元素来表示文本
mpl.rcParams['svg.fonttype'] = 'none'
# 保证同一Matplotlib图形在不同计算机上生成的SVG文件HASH值相同，从而便于比较和合并文件
mpl.rcParams['svg.hashsalt'] = 'hello'

df = pd.read_excel('open.xlsx', index_col=0)
# print(df.columns[2])

mean_accuracy_matrix = np.zeros([10, 8, 8])

sites = list(range(8))
new_sites = []
for s in sites:
    for combination in combinations(sites, s + 1):
        new_sites.append(list(combination))

channels = ['M1', 'DS', 'Gpe', 'Tha', 'Gpi', 'STN', 'SNR', 'PPN']

for i in range(10):

    times_matrix = np.zeros([8, 8])

    for site_select in new_sites:

        # Name
        name = ''
        for n in site_select:
            name = name + channels[n] + '+'
        name = name[:-1]

        # Data
        temp = df[df['Name'] == name]

        for j in site_select:
            # if temp['Site Number'].item() != 8:
            mean_accuracy_matrix[i][j, temp['Site Number'] - 1] = mean_accuracy_matrix[i][j, temp['Site Number'] - 1] + temp[df.columns[i+2]]
            times_matrix[j, temp['Site Number'] - 1] = times_matrix[j, temp['Site Number'] - 1] + 1

    # Compute
    mean_accuracy_matrix[i] = mean_accuracy_matrix[i] / times_matrix

mean = np.mean(mean_accuracy_matrix, axis=0)
std = np.std(mean_accuracy_matrix, axis=0)

dfm = pd.DataFrame(std)
dfm.to_excel('位点组合.xlsx')

# Figure
fig, ax = plt.subplots(figsize=(10, 8))

# X-Label
x_labels = np.array(channels)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

# Plot
for i in range(8):
    for j in range(8):
        ax.plot((j+1, j+1), (mean[i, j]+std[i, j], mean[i, j]-std[i, j]), color=colors[i], alpha=0.7)
        ax.plot((j-0.03+1, j+0.03+1), (mean[i, j]+std[i, j], mean[i, j]+std[i, j]), color=colors[i], alpha=0.7)
        ax.plot((j-0.03+1, j+0.03+1), (mean[i, j]-std[i, j], mean[i, j]-std[i, j]), color=colors[i], alpha=0.7)
    ax.plot(list(range(1, 9)), mean[i], label=x_labels[i], color=colors[i], marker='o', markersize=3, alpha=0.7)
    # ax.scatter(list(range(1, 9)), mean[i], color='white', marker='o', edgecolor=colors[i], zorder=5)


ax.legend(loc='lower right')

# Axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel('Number of Combined Sites')
ax.set_ylabel('Mean Accuracy')

ax.set_ylim(0.3, 0.93)
# fig.savefig('Rats_Mean_Combined_Sites_Accuracy.svg')
