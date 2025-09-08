import numpy as np

from sklearn.preprocessing import scale
from torch.utils.data import Dataset

label_class = {
    'PD': 0,
    'LID': 1,
    'Nor': 2
}


class PsdDataset(Dataset):
    """Make PSD Dataset"""

    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def make_label(data, label):
    """Give Labels"""
    length = data.shape[0]
    labels = []
    labels.extend([label_class[label]] * length)
    return labels


def load_data(data_path):
    """Load Data and Preprocessing"""
    # Load
    data = np.load(data_path)

    # Scale
    for i in range(data.shape[0]):
        data[i] = scale(data[i], axis=1)

    return data


def data_reshape(data, site):
    """Reshape Data"""
    a, b, c = data[:, site, :].shape
    reshaped_data = data[:, site, :].reshape(a, b * c)
    return reshaped_data, b * c

