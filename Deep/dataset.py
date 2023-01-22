import os
import torch
from torch.utils.data import Dataset

class UTRData(Dataset):
    def __init__(self, data, label):
        "Input format should be id-converted sequences"
        super().__init__()
        self.data = data
        self.label = label
    
    def __getitem__(self,index):
        "output format should be id-converted sequences"
        return self.data[index],self.labels[index]
        

    def __len__(self):
        return self.data.shape[0]
