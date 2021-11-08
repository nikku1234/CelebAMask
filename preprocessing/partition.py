# https: // github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_partition.py
import os
import shutil
import pandas as pd
from shutil import copyfile
# from utils import make_folder


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


#### source data path
s_label = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/mask'
s_img = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebA-HQ-img'
s_label_color = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/mask_color'
#### destination training data path
d_train_label = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/train_label'
d_train_img = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/train_img'
d_train_label_color = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/train_label_color'
#### destination testing data path
d_test_label = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/test_label'
d_test_img = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/test_img'
d_test_label_color = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/test_label_color'

#### val data path
d_val_label = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/val_label'
d_val_img = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/val_img'
d_val_label_color = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/val_label_color'

#### make folder
make_folder(d_train_label)
make_folder(d_train_img)
make_folder(d_train_label_color)
make_folder(d_test_label)
make_folder(d_test_img)
make_folder(d_test_label_color)
make_folder(d_val_label)
make_folder(d_val_img)
make_folder(d_val_label_color)

#### calculate data counts in destination folder
train_count = 0
test_count = 0
val_count = 0

image_list = pd.read_csv('/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt',
                         delim_whitespace=True, header=None)
f_train = open(
    '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/train_list.txt', 'w')
f_val = open(
    '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/val_list.txt', 'w')
f_test = open(
    '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data/test_list.txt', 'w')

for idx, x in enumerate(image_list.loc[:, 1]):
    print(idx, x)
    x = int(x)
    if x >= 162771 and x < 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'),
                 os.path.join(d_val_label, str(val_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'),
                 os.path.join(d_val_img, str(val_count)+'.jpg'))
        copyfile(os.path.join(s_label_color, str(idx)+'.png'),
                 os.path.join(d_val_label_color, str(val_count)+'.png'))
        val_count += 1
        f_val.write(str(idx)+'.png'+'\n')

    elif x >= 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'),
                 os.path.join(d_test_label, str(test_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'),
                 os.path.join(d_test_img, str(test_count)+'.jpg'))
        copyfile(os.path.join(s_label_color, str(idx)+'.png'),
                 os.path.join(d_test_label_color, str(test_count)+'.png'))
        test_count += 1
        f_test.write(str(idx)+'.png'+'\n')
    else:
        copyfile(os.path.join(s_label, str(idx)+'.png'),
                 os.path.join(d_train_label, str(train_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'),
                 os.path.join(d_train_img, str(train_count)+'.jpg'))
        copyfile(os.path.join(s_label_color, str(idx)+'.png'),
                 os.path.join(d_test_label_color, str(test_count)+'.png'))
        train_count += 1
        f_train.write(str(idx)+'.png'+'\n')

print(train_count + test_count + val_count)
#### close the file
f_train.close()
f_val.close()
f_test.close()
