import glob2
root_dir = '/home/nramesh8/Desktop/Vision/CelebAMask/CelebAMask-HQ/data'
train_image_path = glob2.glob(root_dir+'/train_img/*.jpg',recursive=True)
train_label_path = glob2.glob(root_dir+'/train_label/*.png',recursive=True)

val_image_path = glob2.glob(root_dir+'/val_img/*.jpg',recursive=True)
val_lable_path = glob2.glob(root_dir+'/val_label/*.png',recursive=True)

test_image_path = glob2.glob(root_dir+'/test_img/*.jpg',recursive=True)
test_label_path = glob2.glob(root_dir+'/test_label/*.png',recursive=True)

print(test_label_path)


