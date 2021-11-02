import os
import re

mapping = "/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"
with open(mapping) as f:
    lines_mapping = f.readlines()
idx_list = []
orig_idx_list = []
orig_file_list = []
final = {}
for i in range(len(lines_mapping)):
    remove_space = re.sub("\s\s+", " ", lines_mapping[i])
    idx, orig_idx, orig_file = remove_space.strip().split(" ")
    