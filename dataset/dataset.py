import numpy as np
import pandas as pd
from wav2mfcc import mp3_to_mfcc
import torch
from torch.utils.data import Dataset

class TuneDataset(Dataset):

    def __init__(self, csv, duration):
        csv_data = pd.read_csv(csv, header=0)
        self.data_size = len(csv_data)
        self.data = np.array((self.data_size, 2))
        for index in range(self.data_size):
            self.data[index][0] = mp3_to_mfcc(csv_data[index][0], csv_data[index][1], csv_data[index][2])
            self.data[index][1] = csv_data[index][3]

    def __len__(self):
        return self.data_size

    def __getitem(self, ix):
        
