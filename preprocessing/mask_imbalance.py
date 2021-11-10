import glob2
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'
train_label_path = glob2.glob(root_dir + '/train_label/*.png', recursive=True)
trans = transforms.Compose([transforms.ToTensor()])
classes = list(range(19))
class_pixel_count = [0] * 19
count = 0
for train_lab in train_label_path:
    label = Image.open(train_lab)
    label = trans(label)
    label *= 255
    label = label.long()
    count += 1
    for class_val in classes:
        class_map = torch.where(label == class_val, 1, 0)
        class_pixel_count[class_val] += torch.sum(class_map)
print(torch.Tensor(class_pixel_count)/count)
