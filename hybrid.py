import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    img_height = len(img)
    img_width = len(img[0])
    kernel_height = (len(kernel)-1)/2
    kernel_width = (len(kernel[0])-1)/2
    
    #result img
    res_img = np.copy(img)
    
    #if the input image is greyscale
    if img.ndim == 2:
        #apply cross correlation for each grid
        for i in range(img_height):
            for j in range(img_width):
                curr_sum = 0
                #calculate weighted sum under the current kernel
                for u in range(-kernel_height,kernel_height+1):
                    for v in range(-kernel_width,kernel_width+1):
                        if i+u >= 0 and i+u < img_height and j+v >= 0 and j+v < img_width: 
                            curr_sum += kernel[u+kernel_height][v+kernel_width]*img[i+u][j+v]
                            
                #update the result img
                res_img[i][j] = curr_sum
                

    #if the input image is rgb
    else:
        #apply cross correlation for each grid
        for i in range(img_height):
            for j in range(img_width):
                r_sum = 0
                g_sum = 0
                b_sum = 0
                #calculate weighted sum under the current kernel
                for u in range(-kernel_height,kernel_height+1):
                    for v in range(-kernel_width,kernel_width+1):
                        if i+u >= 0 and i+u < img_height and j+v >= 0 and j+v < img_width: 
                            r_sum += kernel[u+kernel_height][v+kernel_width]*img[i+u][j+v][0]
                            g_sum += kernel[u+kernel_height][v+kernel_width]*img[i+u][j+v][1]
                            b_sum += kernel[u+kernel_height][v+kernel_width]*img[i+u][j+v][2]
                            
                #update the result img
                res_img[i][j][0] = r_sum
                res_img[i][j][1] = g_sum
                res_img[i][j][2] = b_sum
    # TODO-BLOCK-END
    
    return res_img

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return cross_correlation_2d(img,np.flipud(np.fliplr(kernel)))
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # Generate a zero kernel
    gaussian_filter = np.zeros((height, width))

    # Cenerate the kernel from Lecture notes
    for i in range(-int(height/2), int(height/2) + 1):
        for j in range(-int(width/2), int(width/2) + 1):
            p1 = 1 / (2*np.pi*(math.pow(sigma,2)))
            p2 = math.pow(np.e, -((math.pow(i,2) + math.pow(j,2))/(2 * (sigma*sigma))))
            gaussian_filter[i + int(height/2)][j + int(width/2)] = p1 * p2

    # Normalize the matrix
    gaussian_filter = gaussian_filter/np.sum(gaussian_filter)

    return gaussian_filter

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Low pass = original image - low pass image
    return np.subtract(img, low_pass(img, sigma, size))

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


