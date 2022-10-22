# Author: Adam Kutchak
# canny_edge.py
# implementation of canny edge detection

# importing libs
import cv2
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# CreateGaussian function
#   create a 1 dimensional gaussian kernel from a given size and standard dev.
#   
#   inputs
#       size - size of kernel (default is 3x1)
#       sigma - standard deviation
#
#   output
#       numpy array that represents the gaussian function 
#           in horizontal direction
#       numpy array that represents the gaussian function 
#           in horizontal direction
#
###############################################################################
def CreateGaussian(size, sigma):
    kernel_radius = size // 2
    kernel = np.zeros(2*kernel_radius+1)

    # we need a summation to normalize the weights
    norm_sum = 0

    # apply the Gaussian equation to each value in the kernel to obtain weights
    for i in range(-kernel_radius, kernel_radius+1):

        # Gaussian equation
        weight = 1/(sigma * np.sqrt(2 * np.pi))*np.exp(-(0.5*i**2/sigma**2))

        # update the sum
        norm_sum += weight
        kernel[kernel_radius+i] = weight

    # normalization of the weights
    kernelx = kernel/norm_sum
    # reshape this kernel so it can be applied vertically
    kernely = np.reshape(kernelx, (1, kernel_size))

    return kernelx,kernely

###############################################################################
# CreateGaussianDerivative function
#   create a 1 dimensional kernel for the first derivative of Gaussian
#
#   inputs
#       size - size of kernel (default is 3x1)
#       sigma - standard deviation
#
#   output
#       numpy array that represents a 1st order derivative of gaussian function
#           in the horizontal direction
#       numpy array that represents a 1st order derivative of gaussian function
#           in the vertical direction
#
###############################################################################
def CreateGaussianDerivative(size, sigma):
    kernel,kernely = CreateGaussian(size, sigma)
    kernel_radius = size // 2

    # apply the first derivative Gaussian equation to each value in the kernel
    for i in range(-kernel_radius, kernel_radius+1):

        # First order derivative Gaussian equation
        weight = -i/(sigma**2 * kernel[kernel_radius + i])
        kernel[kernel_radius+i] = -weight
        kernel[kernel_radius-i] = weight


    # reshape this kernel so it can be applied vertically
    kernely = np.reshape(kernel, (1, size))
    return kernel, kernely

###############################################################################
# Convolve Function
#   convolve image with 1 dimensional kernel
#
#   inputs
#       image - input numpy array of an image
#       kernelx - the kernel you wish to convolve with the image horizontally
#       kernely - the kernel you wish to convolve with the image vertically
#
#   output
#       numpy array that is the horizontal convolution of image and kernel
#       numpy array that is the vertical convolution of image and kernel
#
###############################################################################
def Convolve(image, kernelx, kernely):
    # since our kernel is only 1 dimension, it only has a width, no height.
    kernel_size = len(kernelx)
    image_row, image_col = image.shape

    # create a new array to fill out output with
    outputx = np.zeros(image.shape)

    # create some padding for the image so our kernel can convolve nicely
    padhx = kernel_size//2
    padwx = padhx
    
    # we add padding all around 
    # since we're convolving in 1D but both vertically and horizontally
    padi = np.zeros((image_row + 2*padhx, image_col + 2*padwx))
    padi[padhx:padi.shape[0] - padhx, padwx:padi.shape[1] - padwx] = image

    # slide the kernel along the rows and multiply the kernel with values
    for i in range(image_row):
        for j in range(image_col):
            outputx[i, j] = np.sum(kernelx * padi[i: i+kernel_size, j])

    # we need another output for othe y direction
    outputy = np.zeros(image.shape)

    # slide the kernel down the columns and multiply the kernel with values
    for i in range(image_row):
        for j in range(image_col):
            outputy[i, j] = np.sum(kernely * padi[i, j: j+kernel_size])

    return outputx, outputy

###############################################################################
# ComputeMagnitude Function
#   compute the magnitude of the edges in the x and y fields by taking
#   squareroot(x*x + y*y)
#
#   inputs
#       image_horz - numpy array that contains horizontal edge responses
#       image_vert - numpy array that contains vertical edge responses
#
#   output
#       numpy array that is the combination of othe x and y repsonses
#
###############################################################################
def ComputeMagnitude(image_horz, image_vert):
    # both images are the same size, create a new array to fill with our output
    output = np.zeros(image_horz.shape)
    image_row, image_col = image_horz.shape

    # we need the angle of the gradient at each pixel to determine where a
    # local maximum is. We can do this by looking at the 8 neighbors
    angle = np.rad2deg(np.arctan2(image_horz, image_vert))

    # iterate along the image and apply the magnitude equation to each value
    for i in range(image_row):
        for j in range(image_col):
            output[i, j] = np.sqrt(image_horz[i, j]**2 + image_vert[i, j]**2)

    # normalization
    output = output * (255/output.max())
    return output,angle

###############################################################################
# NMS Function
#   apply non maximum suppression to and image
#
#   inputs
#       image - numpy array you wish to apply suppression too 
#
#   output
#       numpy array that has been suppressed with NMS
#
###############################################################################
def NMS(image, angle):
    image_row, image_col = image.shape

    # create an empty array to fill with our output
    output = np.zeros((image_row, image_col))

    # We only need to check 0 - 180 degrees if we remove the negative angles
    # by adding 180 to them
    angle[angle < 0] += 180

    # iterate through the image, check if 
    for i in range(1, image_row-1):
        for j in range(1, image_col-1):

            # q and r are the neighboring pixels that we check depedning on 
            # each of the angles
            q = 255
            r = 255
            
            # check pixels above and below
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = image[i, j+1]
                r = image[i, j-1]

            # check diagonal pixels
            elif (22.5 <= angle[i, j] < 67.5):
                q = image[i+1, j-1]
                r = image[i-1, j+1]

            # check pixels to left and right
            if (67.5 <= angle[i, j] < 112.5):
                q = image[i+1, j]
                r = image[i-1, j]

            # check pixels to left and right
            if (112.5 <= angle[i, j] <= 157.5):
                q = image[i+1, j]
                r = image[i-1, j]

            # check if neighbor is a maximum, if not, remove it
            if (image[i, j] >= q) and (image[i, j] >= r):
                output[i, j] = image[i, j]
            else:
                output[i, j] = 0
    return output

###############################################################################
# Hysteresis Function
#   apply a threshold limit to determine real edges and non-edges values above
#   are real edges, values below are not edges, values in between are validated
#   by looking at their neighbors
#
#   inputs
#       image - numpy array of image
#       high  - values above this threshold limit are real edges
#       low - values below this threshold limit are non-edges
#
#   output
#       numpy array with thresholds applied
#
###############################################################################
def HysteresisThresholding(image, high, low):
    image_row, image_col = image.shape

    # create empty array to fill our output with
    output = np.zeros((image_row, image_col))

    # strong horizontal edges and strong vertical edges
    strong_edge_x, strong_edge_y = np.where(image >= high)

    # vertical and horizontal definite non-edges
    non_edge_x, non_edge_y = np.where(image < low)

    # vertical and horizontal weak edges, we need to check their neighbors
    weak_edge_x, weak_edge_y = np.where((image < high) & (image >= low))

    output[strong_edge_x, strong_edge_y] = 255
    output[non_edge_x, non_edge_y] = 0

    # assign weak edges to some arbitrary values not 0 or 255
    output[weak_edge_x, weak_edge_y] = 100

    # iterate through the image, and check the weak edges' neighbors
    # if they are tangential to a strong edge, then it is an edge,
    # otherwise they are not an edge
    for i in range(1, image_row-1):
        for j in range(1, image_col-1):
            if (output[i, j] == 100):
                # checking if any of the 8 neighbors are strong edges
                if (255 in [output[i-1, j-1], output[i-1, j], output[i-1, j+1],
                    output[i, j-1], output[i, j+1],
                    output[i+1, j-1], output[i+1, j], output[i+1, j+1]]):
                    output[i, j] = 255
                else:
                    output[i, j] = 0

    return output

if __name__ == '__main__':
    # import image with opencv
    I = cv2.imread('data/ostrich.jpg', cv2.IMREAD_GRAYSCALE)
    kernel_size = 5
    sigma = 0.5

    # create gaussian mask
    Gx, Gy = CreateGaussian(kernel_size, sigma)


    # convolve w/ gaussian
    Ix, Iy = Convolve(I, Gx, Gy)
    # save images
    cv2.imwrite('data/01.jpg', Ix)
    cv2.imwrite('data/02.jpg', Iy)

    # create gaussian derivative
    Gxx, Gyy = CreateGaussianDerivative(kernel_size, sigma)

    # convolve with derivatives of gaussian
    Ixx, Iyy = Convolve(Ix, Gxx, Gyy)
    # save images
    cv2.imwrite('data/03.jpg', Ixx)
    cv2.imwrite('data/04.jpg', Iyy)

    # compute magnitude
    M, angle = ComputeMagnitude(Ixx, Iyy)
    # save image
    cv2.imwrite('data/05.jpg', M)

    # non-maximal suppression
    I_nms= NMS(M, angle)
    # save image
    cv2.imwrite('data/06.jpg', I_nms)

    # hysteresis thresholding
    I_hys = HysteresisThresholding(I_nms, 50, 15)
    # save image
    cv2.imwrite('data/07.jpg', I_hys)

