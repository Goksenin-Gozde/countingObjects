import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data,io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


image = data.coins()[10:-80, 10:-80]

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(1))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)
i = 0
for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 50:

        i += 1
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.title("Number of coins : " + str(i))
plt.tight_layout()
plt.show()
