# import glob2
# root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'
# train_image_path = glob2.glob(root_dir+'/train_img/*.jpg',recursive=True)
# train_label_path = glob2.glob(root_dir+'/train_label/*.png',recursive=True)

# val_image_path = glob2.glob(root_dir+'/val_img/*.jpg',recursive=True)
# val_lable_path = glob2.glob(root_dir+'/val_label/*.png',recursive=True)

# test_image_path = glob2.glob(root_dir+'/test_img/*.jpg',recursive=True)
# test_label_path = glob2.glob(root_dir+'/test_label/*.png',recursive=True)

# print(test_label_path)

# from .utils import load_state_dict_from_url
import os
import glob
from posixpath import join
from re import split
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import argparse
from torchvision.transforms.transforms import RandomAffine, RandomVerticalFlip
from dataLoader.celebDataset import celebDatasetTrain, celebDatasetVal, celebDatasetTest
from model.unet import unet
from matplotlib import pyplot as plt
from torchsummary import summary
import cv2

# transformation = transforms.Compose([transforms.ToTensor(),
#                                      transforms.RandomAffine(
#     degrees=(-30, 30), translate=(0.1, 0.1), shear=(0.2)),
#     transforms.RandomHorizontalFlip(0.3),
#     transforms.RandomGrayscale(0.3),
#     transforms.RandomVerticalFlip(0.3),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'

# checkpoint_dir = './checkpoints/model1/'
# if not os.path.exists(checkpoint_dir):
#     os.mkdir(checkpoint_dir)

# load_checkpoint_path = None

# celebDataset_train = celebDatasetTrain(root_dir, transformation)
# train_loader = DataLoader(celebDataset_train, shuffle=True, num_workers=6)
# print(train_loader[0])

print(cv2.imread('/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/test_label/0.png').shape)
