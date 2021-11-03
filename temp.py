import os
import re
import pandas as pd


# mapping = "/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"
# with open(mapping) as f:
#     lines_mapping = f.readlines()
# idx_list = []
# orig_idx_list = []
# orig_file_list = []
# final = {}
# for i in range(len(lines_mapping)):
#     remove_space = re.sub("\s\s+", " ", lines_mapping[i])
#     idx, orig_idx, orig_file = remove_space.strip().split(" ")
#     final[orig_file] = {'idx': idx,
#                        'orig_idx': orig_idx, 'orig_file': orig_file}
#     # final.update(dict)
# print(final)





# attribute_file = "/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
# with open(attribute_file) as f:
#     attribute_mapping = f.readlines()
# columns = attribute_mapping[1].strip().split(" ")
# columns.insert(0,"Image_name")
# columns.insert(1,"Temp_Space")
# data = []
# for row in range(2,len(attribute_mapping)): 
#     data.append(attribute_mapping[row].strip().split(" "))
# print(len(columns))
# new = pd.DataFrame(data,columns=columns)
# print(new.head())


# columns = attribute_mapping[1].strip().split(" ")
# data = []
# for row in range(2, len(attribute_mapping)):
#     data.append(attribute_mapping[row].strip().split(" ")[2:])
# print(len(columns))
# new = pd.DataFrame(data, columns=columns)
# print(new.head())




pose_file = "/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/CelebAMask-HQ-pose-anno.txt"
with open(pose_file) as f:
    pose_mapping = f.readlines()

columns = pose_mapping[1].strip().split(" ")
print(columns)
# print(pose_mapping[2].strip().split(" ")[1:])

pose_data = []
for row in range(2, len(pose_mapping)):
    pose_data.append(pose_mapping[row].strip().split(" ")[1:])
print(len(columns))
new = pd.DataFrame(pose_data, columns=columns)
print(new.head())
