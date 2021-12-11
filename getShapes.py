from PIL.Image import new
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os

pattern = 'Robe_jacket_2.png'
new_h = 1000
scale = 0  # pixels per inch

big_out_dir = pattern.split('.')[0]+"_pattern_shapes"
shape_dir = big_out_dir + "/pattern_shapes"
rotate_dir = big_out_dir+'/rotated_shapes'

if not os.path.isdir(big_out_dir):
    os.mkdir(big_out_dir)
    os.mkdir(shape_dir)
    os.mkdir(rotate_dir)

pattern_img1 = cv2.imread(pattern)
r = new_h/pattern_img1.shape[0]
dim = (int(pattern_img1.shape[1]*r), 1000)
resized = cv2.resize(pattern_img1, dim, interpolation=cv2.INTER_AREA)

pattern_img = cv2.cvtColor(pattern_img1, cv2.COLOR_RGB2GRAY)
pattern_img = cv2.bitwise_not(pattern_img)


ret, thresh = cv2.threshold(pattern_img, 50, 255, 0)

# opened = np.zeros(thresh.shape)
dilation_kernel = np.ones((4, 4), np.uint8)
closing_kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, dilation_kernel, iterations=1)
# closed = np.zeros(thresh.size)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,
                          closing_kernel, iterations=5)
# plt.imshow(closed), plt.show()


# thank you to https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# for this


contours, hierarchy = cv2.findContours(
    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(pattern_img1.shape, np.uint8)
mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)


mask_thresh = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)


(retval, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask_thresh)
print(len(labels), len(stats), len(centroids))
output = pattern_img.copy()
numLabels = len(stats)

bw_pattern = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

pattern_shapes = []

px_per_inch = int(input("How many pixels per inch: "))
fabric_width = int(input("How many inches wide is the fabric:"))
fabric_length = int(input("How many inches long is the fabric:"))

CANVAS_PX_PER_IN = 15  # to resize the images

# loop over the number of unique connected component labels
for i in range(0, numLabels):
    # if this is the first component then we examine the
    # *background* (typically we would just ignore this
    # component in our loop)
    if i == 0:
        text = "examining component {}/{} (background)".format(
            i + 1, numLabels)
        continue
    # otherwise, we are examining an actual connected component
    else:
        text = "examining component {}/{}".format(i + 1, numLabels)
    # print a status message update for the current connected
    # component
    print("[INFO] {}".format(text))
    # extract the connected component statistics and centroid for
    # the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    if area > 22000:
        mask_rec = mask[y:y+h, x:x+w]
        pattern_shape = cv2.cvtColor(
            pattern_img1[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        componentMask = (labels == i).astype("uint8") * 255
        componentMask = componentMask[y:y+h, x:x+w]
        # bw_pattern_shape = cv2.cvtColor(pattern_shape, cv2.COLOR_RGB2GRAY)
        componentMask = cv2.cvtColor(
            componentMask.astype('uint8'), cv2.COLOR_GRAY2RGB)
        new_shape = cv2.bitwise_not(pattern_shape*componentMask)
        # tmp = cv2.cvtColor(new_shape, cv2.COLOR_BGR2GRAY)
        # _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY_INV)
        # b, g, r = cv2.split(new_shape)
        # rgba = [b, g, r, alpha]
        # dst = cv2.merge(rgba, 4)
        # (H, W) = dst.shape[:2]
        # newH = int(H*(CANVAS_PX_PER_IN/px_per_inch))
        # newW = int(W*(CANVAS_PX_PER_IN/px_per_inch))
        # dst = cv2.resize(dst, (newW, newH))
        # cv2.imwrite(shape_dir+"/new_shape_transparent"+str(i)+".png", dst)
        cv2.imwrite(shape_dir+"/new_shape"+str(i)+".png", new_shape)
        # pattern_shapes.append(new_shape)
        # plt.imshow(new_shape, cmap="gray_r")
        # plt.show()
        # # plt.imsave(shape_dir+"/new_shape"+str(i)+".png", new_shape)
        # plt.imsave(shape_dir+"/new_shape_transparent"+str(i)+".png", dst)


# plt.imshow(pattern_img1)
# # plt.imsave("mask.png", componentMask)
plt.show()

# turn things into svgs
# https://stackoverflow.com/questions/43108751/convert-contour-paths-to-svg-paths
