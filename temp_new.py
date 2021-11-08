from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
import matplotlib
# Since you are generating a random colormap, it's no surprise the colors are not consistent.

# You want to create one colormap with your colors of choice, and pass that colormap to each image, regardless of the number of blobs present. Note however, that by default the colormaps are normalized to the range of your data. Since the range of the data changes depending on the number of blobs found, you need to explicitly set the normalization using vmin = and vmax. Here is a demonstration using 4 different images:


colors = ['black', 'red', 'green', 'yellow', 'pink', 'orange']
vmin = 0
vmax = len(colors)
cmap = matplotlib.colors.ListedColormap(colors)


fig, axs = plt.subplots(4, 1, figsize=(3, 3*4))
for file, ax in zip(['./CelebAMask-HQ/mask/0.png', './CelebAMask-HQ/mask/1.png', './CelebAMask-HQ/mask/2.png', './CelebAMask-HQ/mask/3.png'], axs):
    im = imageio.imread(file)
    blobs, number_of_blobs = ndimage.label(im)
    ax.imshow(blobs, cmap=cmap, vmin=vmin, vmax=vmax)
