from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
import glob2
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class celebDatasetTrain(Dataset):
    def __init__(self, root_dir, transform, target_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train_image_path = glob2.glob(root_dir+'/train_img/*.jpg', recursive=True)
        self.train_label_path = glob2.glob(root_dir+'/train_label/*.png', recursive=True)

    def __len__(self):
        return len(self.train_image_path)

    def __getitem__(self,idx):

        image = cv2.imread(self.train_image_path[idx])
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        # label = cv2.imread(self.train_label_path[idx])
        label = Image.open(self.train_label_path[idx])
        image = self.transform(image)
        label = self.target_transform(label)

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

        # self.train_image_path = glob2.glob(
        #     root_dir+'/train_img/*.jpg', recursive=True)
        # self.train_label_path = glob2.glob(
        #     root_dir+'/train_label/*.png', recursive=True)

        self.val_image_path = glob2.glob(
            root_dir+'/val_img/*.jpg', recursive=True)
        self.val_label_path = glob2.glob(
            root_dir+'/val_label/*.png', recursive=True)

        # self.test_image_path = glob2.glob(
        #     root_dir+'/test_img/*.jpg', recursive=True)
        # self.test_label_path = glob2.glob(
        #     root_dir+'/test_label/*.png', recursive=True)

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

        # self.train_image_path = glob2.glob(
        #     root_dir+'/train_img/*.jpg', recursive=True)
        # self.train_label_path = glob2.glob(
        #     root_dir+'/train_label/*.png', recursive=True)

        # self.val_image_path = glob2.glob(
        #     root_dir+'/val_img/*.jpg', recursive=True)
        # self.val_lable_path = glob2.glob(
        #     root_dir+'/val_label/*.png', recursive=True)

        self.test_image_path = glob2.glob(
            root_dir+'/test_img/*.jpg', recursive=True)
        self.test_label_path = glob2.glob(
            root_dir+'/test_label/*.png', recursive=True)

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

        final = {
            "image": image,
            "label": label
        }
        return final

