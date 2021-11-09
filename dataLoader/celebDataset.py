from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
import glob2
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

class celebDatasetTrain(Dataset):
    def __init__(self, root_dir, transform, target_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train_image_path = glob2.glob(root_dir+'/train_img/*.jpg', recursive=True)
        # self.train_label_path = glob2.glob(root_dir+'/train_label/*.png', recursive=True)
        self.train_label_path = []
        for val in self.train_image_path:
            fname = val.split('/')[-1]
            fname = fname.split('.')[0]
            self.train_label_path.append(root_dir + '/train_label/' + fname + '.png')

    def __len__(self):
        return len(self.train_image_path)

    def __getitem__(self,idx):
        # torch.set_printoptions(profile="full")
        image = cv2.imread(self.train_image_path[idx])
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        # label = cv2.imread(self.train_label_path[idx])
        label = Image.open(self.train_label_path[idx])
        image = self.transform(image)
        label = self.target_transform(label)
        label *= 255
        # print(label.min(), label.max())
        label = label.long()
        # print("img:", self.train_image_path[idx], 'label:', self.train_label_path[idx])
        final = {
            "image": image,
            "label": label
        }
        return final
        


class celebDatasetVal(Dataset):
    def __init__(self, root_dir, transform, target_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform


        self.val_image_path = glob2.glob(
            root_dir+'/val_img/*.jpg', recursive=True)
        self.val_label_path = []
        for val in self.val_image_path:
            fname = val.split('/')[-1]
            fname = fname.split('.')[0]
            self.val_label_path.append(root_dir + '/val_label/' + fname + '.png')

    def __len__(self):
        return len(self.val_image_path)

    def __getitem__(self, idx):

        image = cv2.imread(self.val_image_path[idx])
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        # label = cv2.imread(self.train_label_path[idx])
        label = Image.open(
            self.val_label_path[idx])
        image = self.transform(image)
        label = self.target_transform(label)
        label *= 255
        label = label.long()
        final = {
            "image": image,
            "label": label
        }
        return final


class celebDatasetTest(Dataset):
    def __init__(self, root_dir, transform, target_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.test_image_path = glob2.glob(
            root_dir+'/test_img/*.jpg', recursive=True)
        # self.test_label_path = glob2.glob(
        #     root_dir+'/test_label/*.png', recursive=True)
        self.test_label_path = []
        for val in self.test_image_path:
            fname = val.split('/')[-1]
            fname = fname.split('.')[0]
            self.test_label_path.append(root_dir + '/test_label/' + fname + '.png')
    def __len__(self):
        return len(self.test_image_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.test_image_path[idx])
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        # label = cv2.imread(self.train_label_path[idx])
        label = Image.open(
            self.test_label_path[idx])
        image = self.transform(image)
        label = self.target_transform(label)
        label *= 255
        label = label.long()
        final = {
            "image": image,
            "label": label
        }
        return final

