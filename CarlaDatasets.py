import os
import torch
from skimage import io
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

#Code from PyTorch Dataset-Documentation: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CarlaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):

        img_path = os.path.join(str(self.img_dir), str(self.img_labels.iloc[index, 0]))
        image = io.imread(img_path)
        image = image[:,:,:3]
        y_label = torch.tensor(int(self.img_labels.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# NPDataset wurde probeweise verwendet um .npy Dateien einzulesen, statt csv-Dateien.
class NPDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = np.load(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #print(int(self.annotations[index, 0]))
        name = str(int(self.annotations[index, 0]))
        padding = "000000000"
        if name[len(name)-2:] == "13":
            padding = "000000000" # eine Null mehr, weil Wetter ID eine Zahl länger ist
        img_path = os.path.join(self.root_dir, f"{padding[0:len(padding)-(len(name))]}{name}.png")
        image = io.imread(img_path)
        image = image[:, :, :3]
        y_label = torch.tensor(int(self.annotations[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
