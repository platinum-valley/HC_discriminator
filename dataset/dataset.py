import numpy as np
import pandas as pd
import pickle as pkl
import torch
import sys
from torch.utils.data import Dataset

class TuneDataset(Dataset):

    def __init__(self, csv, transform=None):
        csv_data = pd.read_csv(csv, header=None)
        self.data_size = len(csv_data)
        self.data = []
        for index in range(self.data_size):
            with open("./dataset/{}".format(csv_data.ix[index, 0]), "rb") as f:
                feat = pkl.load(f)
            feat = torch.tensor(feat)
            feat = torch.transpose(feat, 0, 1)
            if csv_data.ix[index, 1]:
                label = 1.0
            else:
                label = 0.0
            self.data.append([feat, torch.tensor(label)])
        self.sequence_length = len(self.data[0][0].tolist())
        self.feature_num = len(self.data[0][0][0].tolist())
        self.transform = transform

    def __len__(self):
        return self.data_size

    def get_sequence_length(self):
        return self.sequence_length


    def get_feature_num(self):
        return self.feature_num


    def __getitem__(self, idx):
        feature = self.data[idx][0]
        label = self.data[idx][1]
        if self.transform:
            feature = self.transform(feature)
        return (feature, label)


