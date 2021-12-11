
import cv2
import matplotlib.pyplot as plt
from easyocr import Reader
import imutils
from imutils.object_detection import non_max_suppression
import glob
import numpy as np

MIN_CONFIDENCE = 0.4
CANVAS_PX_PER_IN = 35
COUNT = 0


def open_and_resize(img, resize_fact=2):
    # from https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
    # img = "new_shape11.png"
    image = cv2.imread(img)
    orig = image.copy()
    (H, W) = image.shape[:2]

    newW = int(W/resize_fact)
    newH = int(H/resize_fact)

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    return image


# https://www.pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def detect_text(image):
    langs = ['en']
    reader = Reader(langs, gpu=False)
    results = reader.readtext(image)
    return results


def detect_grainline(image, out_dir, img, min_confidence=MIN_CONFIDENCE):
    # langs = ['en']
    # reader = Reader(langs, gpu=False)
    # results = reader.readtext(image)
    grainline_detected = False
    times_rotated = 0
    current_image = image

    # fabric_width = int(input("How many inches wide is the fabric:"))
    # fabric_length = int(input("How many inches long is the fabric:"))

    while (not grainline_detected and times_rotated < 4):
        print(grainline_detected, times_rotated)
        results = detect_text(current_image)
        print("detected text")
        for (bbox, text, prob) in results:
            if "cut" in text:
                print(text)
            if "GRAINLINE" == text:
                grainline_detected = True
                print("grainline detected!")
                break
        if not grainline_detected:
            print("rotate")
            current_image = imutils.rotate_bound(current_image, 90)
            times_rotated += 1

        else:
            break

    if grainline_detected:
        print("Grainline was detected!")
    else:
        print("No grainline detected")
    # loop over the results
    for (bbox, text, prob) in results:
        print(text)
        if text.strip() == 'GRAINLINE':
            # display the OCR'd text and associated probability
            print("[INFO] {:.4f}: {}".format(prob, text))
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            text = cleanup_text(text)
            # cv2.rectangle(current_image, tl, br, (0, 255, 0), 2)
            # cv2.putText(current_image, text, (tl[0], tl[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    plt.imsave(out_dir + "rotated_image" + str(COUNT) + ".png", current_image)

    tmp = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    dilation_kernel = np.ones((4, 4), np.uint8)
    closing_kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(tmp, dilation_kernel, iterations=1)
    _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(current_image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    print(dst.shape[:2], dst.shape)
    (H, W) = dst.shape[:2]
    newH = int(H*2*(CANVAS_PX_PER_IN/px_per_inch))
    newW = int(W*2*(CANVAS_PX_PER_IN/px_per_inch))
    dst = cv2.resize(dst, (newW, newH))

    cv2.imwrite(out_dir+"/rotated_transparent_" +
                str(COUNT)+".png", dst)

    # # show the output image
    # plt.imshow(current_image)
    # plt.imsave(img.split('.')[0] + "rotated_image.png", current_image)
    # plt.show()


in_dir = "Robe_jacket_1_pattern_shapes/pattern_shapes"
out_dir = "Robe_jacket_1_pattern_shapes/rotated_shapes"
px_per_inch = int(input("How many pixels per inch: ") or 145)
for img in glob.glob(f"{in_dir}/*.png"):
    print("Detecting grainline in", img)
    image = open_and_resize(img)
    detect_grainline(image, out_dir, img)
    COUNT += 1
