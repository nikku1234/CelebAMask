import glob2
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'
train_label_path = glob2.glob(root_dir + '/train_label/*.png', recursive=True)
trans = transforms.Compose([transforms.ToTensor()])
for train_lab in train_label_path:
    label = Image.open(train_lab)
    # label = self.target_transform(label)
    label = trans(label)
    label *= 255
    label = label.long()
# class celebDatasetTrain(Dataset):
#     def __init__(self, root_dir, transform, target_transform=None) -> None:
#         super().__init__()
#         self.root_dir = root_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.train_image_path = glob2.glob(root_dir+'/train_img/*.jpg', recursive=True)
#         # self.train_label_path = glob2.glob(root_dir+'/train_label/*.png', recursive=True)
#         self.train_label_path = []
#         for val in self.train_image_path:
#             fname = val.split('/')[-1]
#             fname = fname.split('.')[0]
#             self.train_label_path.append(root_dir + '/train_label/' + fname + '.png')

#     def __len__(self):
#         return len(self.train_image_path)

#     def __getitem__(self,idx):
#         # torch.set_printoptions(profile="full")
#         image = cv2.imread(self.train_image_path[idx])
#         # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
#         image = Image.fromarray(image)
#         # label = cv2.imread(self.train_label_path[idx])
#         label = Image.open(self.train_label_path[idx])
#         image = self.transform(image)
#         label = self.target_transform(label)
#         label *= 255
#         # print(label.min(), label.max())
#         label = label.long()
#         # print("img:", self.train_image_path[idx], 'label:', self.train_label_path[idx])
#         final = {
#             "image": image,
#             "label": label
#         }
#         return final
