# https: // github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py
import os.path as osp
import os
import cv2
import numpy as np
from PIL import Image

face_data = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/new_mask'
counter = 0
total = 0
for i in range(15):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = []

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)
            new = np.empty((512,512), dtype=float, order='C')
            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                print(sep_mask.shape)
                print(sep_mask.shape)
                new = np.stack((new, sep_mask), axis=0)
                # print(np.unique(sep_mask))
                # new = np.stack(sep_mask, axis=0).shape
                # mask[sep_mask == 225] = l
        print(new.shape)
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(j)

print(counter, total)
