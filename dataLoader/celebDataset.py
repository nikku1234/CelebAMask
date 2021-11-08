from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class celebDataset(Dataset):
    def __init__(self,mapping_path,attribute_path,pose_path) -> None:
        super().__init__()
        self.mapping = mapping_path
        self.attribute = attribute_path
        self.pose = pose_path
        
        with open(self.mapping) as f:
            lines_mapping = f.readlines()
        
        with open(self.attribute) as f:
            attribute_mapping = f.readlines()
        
        with open(self.pose) as f:
            pose_mapping = f.readlines()

    
    def __len__(self):
        return len(self)

    def __getitem__(self,idx):

        pass

