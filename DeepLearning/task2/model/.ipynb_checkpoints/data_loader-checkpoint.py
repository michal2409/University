from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import glob

class CityscapesDataset(Dataset):
    def __init__(self, path, augment=None):
        self.path = path
        self.filenames = [os.path.basename(os.path.normpath(fname)) for fname in glob.glob(self.path + '/X/*.png')]
        self.augment = augment
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        X = plt.imread(self.path + '/X/' + self.filenames[index]) * 255
        Y = plt.imread(self.path + '/Y/' + self.filenames[index]) * 255
        label = plt.imread(self.path + '/labeled/' + self.filenames[index]) * 255
        
        if self.augment:
            X, label = self.augment(X, label)
        
        X = self.toTensor(X)
        
        return X, Y, label, self.filenames[index]
    
def fetch_dataloader(data_dir, batch_size, augment=None):
    dataset = CityscapesDataset(data_dir, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return dataloader
