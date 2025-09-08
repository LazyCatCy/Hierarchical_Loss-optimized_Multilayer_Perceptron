import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
import seaborn as sns
import os


mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 14
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

channels = ['M1', 'DS', 'Gpe', 'Tha', 'Gpi', 'STN', 'SNR', 'PPN']

# 加载数据
input_file = r'C:\Users\LazyCat\Desktop\Article\2 Figure\1 Rats\模型画图\位点组合\open.xlsx'
df = pd.read_excel(input_file)

acc_names = ['Open Accuracy', 'Open Accuracy.1', 'Open Accuracy.2', 'Open Accuracy.3', 'Open Accuracy.4',
             'Open Accuracy.5', 'Open Accuracy.6', 'Open Accuracy.7', 'Open Accuracy.8', 'Open Accuracy.9']

information_all = [[], [], [], [], [], [], [], []]

for acc_name in acc_names:
    # 计算数据集熵
    bins = np.linspace(min(df[acc_name]), max(df[acc_name]), 10)
    hist, _ = np.histogram(df[acc_name], bins=bins)
    probabilities = hist / len(df[acc_name])
    dataset_entropy = entropy(probabilities)

    def conditional_entropy(region):
        subsets = [df[df['Name'].str.contains(region)][acc_name], df[~df['Name'].str.contains(region)][acc_name]]
        subset_entropies = [entropy(np.histogram(subset, bins=bins)[0] / len(subset)) if len(subset) > 0 else 0 for subset in subsets]
        total_size = len(df[acc_name])
        weighted_entropy = sum([len(subset) / total_size * ent for subset, ent in zip(subsets, subset_entropies)])
        return weighted_entropy

    # 提取脑区
    brain_regions = set([region for combination in df['Name'] for region in combination.split('+')])

    # 计算所有脑区的条件熵
    conditional_entropies = {region: conditional_entropy(region) for region in brain_regions}

    # 创建一个字典来存储信息增益
    information_gains = {region: dataset_entropy - ent for region, ent in conditional_entropies.items()}

    for i in range(8):
        information_all[i].append(information_gains[channels[i]])

information_all = np.array(information_all)
mean = np.mean(information_all, axis=1)
std = np.std(information_all, axis=1)

df2 = pd.DataFrame(information_all)
df2.to_excel('信息熵.xlsx')

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x=channels, height=mean, align='center')
for i in range(8):
    ax.plot([i, i], [mean[i], mean[i]+std[i]], color='black')
    print(mean[i], std[i])
    ax.plot([i-0.4, i+0.4], [mean[i]+std[i], mean[i]+std[i]], color='black')
ax.set_ylabel('Information Gain')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# fig.savefig('InformationGain.svg')
plt.show()
