import numpy as np
import cv2
import os

def preprocess(path):
    """
    Loads images from path into a single-channel numpy array.
    """
    kImageSize = (400, 400)
    images = []
    for file in os.listdir(path):
        print("Reading file: " + file)
        img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
        # Make sure all images are 400x400
        img = cv2.resize(img, kImageSize)
        images.append(img)
    images = np.array(images)
    # Normalize to [0, 1]
    images = images/ 255.0
    # Expand so final array is channels
    images = np.expand_dims(images, len(images.shape))
    print("Processed " + str(images.shape[0]) + " images")
    print("Final shape: " + str(images.shape))
    return images

# test
preprocess("/home/brad/Graphics/CloudGAN/CloudGAN/data/CCSN/unpacked")