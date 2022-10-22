# Author: Adam Kutchak
# image_binarization.py
# implementation of image binarization manually and with otsu

# importing libs
import cv2
import matplotlib.pyplot as plt
import numpy as np

def otsu(path):
    # import image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # collect histogram of the image pixel intensities
    hist, bin_edges = np.histogram(image, bins=256)
    # find center of bins
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    # weight1: sum of probabilities of intensity starting from the front
    w1 = np.cumsum(hist)
    # weight2: sum of probabilities of intensity starting from the back
    w2 = np.cumsum(hist[::-1])[::-1]

    # calculating the mean1 and mean2:
    # mean1: sum of probabilities * intensity values divided by w1
    m1 = np.cumsum(hist * bin_center) / w1
    # mean2: sum of probabilities * intensity values divided by w2
    # starting from the back
    m2 = (np.cumsum((hist * bin_center)[::-1]) / w2[::-1])[::-1]

    # calulating variances
    variance = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:])**2
    # grab the maximum inter class variance
    val = np.argmax(variance)
    # convert float to int for threshold
    threshold = int(bin_edges[val])

    # if the intensity of a pixel is above the threshold, set it to 255
    # otherwise set it to 255
    image[image > threshold] = 255
    image[image <= threshold] = 0
    print(threshold)
    return threshold, image

def image_binarization(path, threshold):
    # import image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # collect histogram of the image pixel intensities
    hist, bins = np.histogram(image, range(0, 257))
    plt.plot(bins[0:-1], hist)
    plt.show()
    # if the intensity of a pixel is above the threshold, set it to 255
    # otherwise set it to 255
    image[image > threshold] = 255
    image[image <= threshold] = 0
    return threshold, image

if __name__ == '__main__':
    # threshold, ostrich = image_binarization('data/ostrich.jpg', 100)
    # cv2.imshow('threshold: '+str(threshold), ostrich)
    # threshold, lady = image_binarization('data/lady.jpg', 150)
    # cv2.imshow('threshold: '+str(threshold), lady)
    # threshold, flower = image_binarization('data/flower.jpg', 70)
    # cv2.imshow('threshold: '+str(threshold), flower)

    threshold, ostrich = otsu('data/ostrich.jpg')
    cv2.imshow('threshold: '+str(threshold), ostrich)
    threshold, lady = otsu('data/lady.jpg')
    cv2.imshow('threshold: '+str(threshold), lady)
    threshold, flower = otsu('data/flower.jpg')
    cv2.imshow('threshold: '+str(threshold), flower)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
