''' Assignment description
This assignment applies variations of Gaussian filter and its derivative as well as Laplacian filter / DoG and Canny edge detection. A description of the application is provided in a separate text file.
'''

# Packages 
from skimage.data import camera
from skimage.io import imread
from skimage.filters import gaussian, difference_of_gaussians, median
from skimage.filters.rank import mean

from skimage import img_as_float
from skimage.io import imshow 
from skimage.util import random_noise
from skimage.morphology import disk
from scipy.signal import correlate2d

# from scalespace import gaussian_scalespace, structure_tensor, determinant_symmetric_field, trace_symmetric_field

import imageio

import cv2
from cv2 import Canny

from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

import numpy as np
import matplotlib.pyplot as plt

import random
import argparse # To make arguments from the terminal
import os



# Gaussian
# create a matrix with gauss distributed values over specified size and sigma
def gaus_matr(size, sigma):
    return np.random.normal(0, sigma, (size, size))

def convolution(img, conv_matrix):
    # apply cross correlation and initialize output matrix with same size as original image
    conv_matrix = np.fliplr(np.flipud(conv_matrix))
    output = np.zeros((img.shape[0], img.shape[1]))
    # fill matrix with convoluted values
    # ignore edges at this step
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not i+conv_matrix.shape[0] > output.shape[0] and not j+conv_matrix.shape[1] > output.shape[1]:
                output[i, j] = (conv_matrix*img[i:i+conv_matrix.shape[0], j:j+conv_matrix.shape[1]]).sum()

    return output

def gaussian_convolutions(lenna,gaus_matr, convolution, output_path):
    plt.rcParams['figure.figsize'] = [12, 12]

    # the bigger the size of the kernel, the more blury and all-grey the image gets
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        con_matr = gaus_matr(4**i, 10)
        fglenna = convolution(lenna, con_matr)
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Size of kernel = {}".format(4**i))
    fig.suptitle('Gaussian Filter with different kernel sizes, sigma = 10')
    plt.savefig(os.path.join(output_path, "GaussianKernelSizes.png"))

    # the bigger the size of the kernel, the more blury and all-grey the image gets
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        con_matr = gaus_matr(8, 2**i)
        fglenna = convolution(lenna, con_matr)
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**i))
    fig.suptitle('Gaussian Filter with different sigma, kernel size = 8')
    plt.savefig(os.path.join(output_path, "GaussianSigmas.png"))

    fig, axes = plt.subplots(2, 2)

    for i in range(4):
        fglenna = gaussian(lenna, sigma=2**(i))
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**(i)))
    fig.suptitle('Gaussian')
    plt.savefig(os.path.join(output_path, "GaussianSkimageSigmas.png"))

    low_res = cv2.resize(lenna, dsize=(128, 128), interpolation=None)
    fig, axes = plt.subplots(2, 2)

    for i in range(4):
        fglenna = gaussian(low_res, sigma=2**(i))
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**(i)))
    fig.suptitle('Gaussian')
    plt.savefig(os.path.join(output_path, "GaussianImgsize.png"))  


    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        fglenna = gaussian_gradient_magnitude(lenna, sigma=2**(i))
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna, cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**(i)))
    fig.suptitle('Gaussian Magnitude');
    plt.savefig(os.path.join(output_path, "GaussianMagnitude_1st.png"))  

    fig, axes = plt.subplots(2, 2)

    for i in range(4):
        fglenna = gaussian_gradient_magnitude(gaussian_gradient_magnitude(gaussian(lenna, sigma=2**(i)), sigma=2**(i)), sigma=2**(i))
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**(i)))
    fig.suptitle('Gaussian Magnitude of Gaussian Magnitude of Gaussian')
    plt.savefig(os.path.join(output_path, "GaussianMagnitude_2nd.png"))  

    fig, axes = plt.subplots(2, 2)

    for i in range(4):
        fglenna = gaussian_gradient_magnitude(gaussian_gradient_magnitude(gaussian_gradient_magnitude(gaussian(lenna, sigma=2**(i)), sigma=2**(i), mode='reflect'), sigma=2**(i), mode='reflect'), sigma=2**(i), mode='reflect')
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna,cmap='Greys_r',interpolation='none')
        ax.set_title("Sigma = {}".format(2**(i)))
    fig.suptitle('Gaussian Magnitude of Gaussian Magnitude of Gaussian Magnitude of Gaussian')
    plt.savefig(os.path.join(output_path, "GaussianMagnitude_3rd.png"))

    
def salt_pepper(lenna, output_path):
    lenna_noise = random_noise(lenna, 'pepper', amount=0.11)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(10, 10)
    for i in range(4):
        sigma = 2**(i)
        fglenna_noise = gaussian(lenna_noise, sigma=sigma)
        ax = axes[i//2, i%2]    
        ax.imshow(fglenna_noise,cmap='Greys_r',interpolation='none')
        ax.set_title(f"Sigma = {sigma}")
    fig.suptitle('Gaussian vs. Salt and Pepper')
    plt.savefig(os.path.join(output_path, "SaltPepperLenna.png"))

def disk_gaussian(output_path):
    d1 = np.array(disk(1))
    D1 = np.zeros((25,25))
    D1[11:14,11:14]=d1
#    plt.imshow(D1,cmap='Greys_r') #small cross to experiment with
#    plt.savefig(os.path.join(output_path, "SmallDisk.png"))
    
    fig, axes = plt.subplots(nrows=2, ncols=2) #plots applying Gaussian filter with increasing sigma value
    fig.set_size_inches(10, 10)
    for i in range(4):
        sigma = 2**(i)
        fgd1 = gaussian(D1, sigma=sigma)
        ax = axes[i//2, i%2]    
        ax.imshow(fgd1,cmap='Greys_r',interpolation='none')
        ax.set_title(f"Sigma = {sigma}")
    fig.suptitle('Gaussian vs. Small Disk')
    plt.savefig(os.path.join(output_path, "DiskGaussian.png"))
    
def DoG(lenna, output_path):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i in range(4):
        sigma = 2**(i)
        diff_gaus_lenna = difference_of_gaussians(lenna, low_sigma = sigma,truncate =3.0)
        ax = axes[i//2, i%2]    
        ax.imshow(diff_gaus_lenna,cmap='Greys_r',interpolation='none')
        ax.set_title(f"Sigma = {sigma}")
    fig.suptitle('Difference of gaussians')
    plt.savefig(os.path.join(output_path, "DoG.png"))
    
def get_barcode_image(bar_width,w = 128, h = 128):
    bars_image = np.zeros((w,h))
    for i in range(0,w):
        if i % bar_width == 0:
            for j in range(int(bar_width/2)):
                bars_image[i+j] = np.ones(w)
    return bars_image
    
# function that plots the difference of gaussians filterd over the barcode image with different sigmas
def plot_diff_gauss_bars(bar_width):
    bars_image = get_barcode_image(bar_width =bar_width,w = 64, h =64 )
    fig, axes = plt.subplots(nrows=3, ncols=2)

    for i in range(6):
        if i ==0:
            axes[0,0].imshow(bars_image,cmap='Greys_r',interpolation='none')
            axes[0,0].set_title('original')
        if i >0:
            sigma = 2**(i-1)
            diff_gaus_bars = difference_of_gaussians(bars_image, low_sigma = sigma,truncate =3.0)
            ax = axes[i//2 , i%2]    
            ax.imshow(diff_gaus_bars,cmap='Greys_r',interpolation='none')
            ax.set_title(f"Sigma = {sigma}")
            fig.tight_layout() 
            fig.subplots_adjust(top=0.88)
    fig.suptitle('Difference of gaussians')
    return fig


# generate image with bars with different shade of greys
def get_grey_bars():
    shaded_bars = np.zeros((256,256))

    counter = 0
    for i in range(0,8):
        for j in range(0,32):
            shaded_bars[counter] = i/4.
            counter +=1
    return shaded_bars
 
# plot shaded bars when filtered with the difference of gaussian with different sigmas
# bars_image = get_barcode_image(bar_width =bar_width,w = 64, h =64 )
def DoG_shaded_bars(output_path):
    shaded_bars = get_grey_bars()
    fig, axes = plt.subplots(nrows=3, ncols=2)

    for i in range(6):
        if i ==0:
            axes[0,0].imshow(shaded_bars, cmap='Greys_r',interpolation='none')
            axes[0,0].set_title('original')
        if i >0:
            sigma = 2**(i-1)
            diff_gaus_bars = difference_of_gaussians(shaded_bars, low_sigma = sigma,truncate =3.0)
            ax = axes[i//2 , i%2]    
            ax.imshow(diff_gaus_bars,cmap='Greys_r',interpolation='none')
            ax.set_title(f"Sigma = {sigma}")
            fig.tight_layout() 
            fig.subplots_adjust(top=0.88)
    fig.suptitle('Difference of gaussians')
    plt.savefig(os.path.join(output_path, "DoG_shaded_bars.png"))

        
# Scale and its effect on Canny edge detection
def scale_canny(cv2lenna, output_path):
    # Scale and threshold values
    scales = [1, 2, 4, 8]
    low = np.percentile(cv2lenna, 5)
    high = np.percentile(cv2lenna, 65)
    # Define axes properties
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax_list = axes.flatten()
    # Going through axes and scale values
    for ax, scale in zip(ax_list, scales):
        # Smoothing
        smoothed = gaussian_filter(cv2lenna,
                            scale, # standard deviation for Gaussian kernel
                            order=0, # An order of 0 corresponds to convolution with a Gaussian kernel. A positive order corresponds to convolution with that derivative of a Gaussian.
                            output=None, # The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.
                            mode='reflect', # ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
                            cval=0.0, # Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
                            truncate=4.0) # Truncate the filter at this many standard deviations. Default is 4.0.
        canny = Canny(smoothed, low, high) 
        ax.imshow(canny);
        ax.set_title(f"Scale: {scale}")

    # Saving figure    
    plt.savefig(os.path.join(output_path, "ScaleCanny.png"))
    print(f"Scale effect on Canny has been computed and saved in {output_path}") 


# Histogram of pixel values
def pixel_histogram(cv2lenna, output_path):
    # Create a vector from image pixels
    flattened = cv2lenna.flatten()
    # Creating figure
    fig, ax = plt.subplots(figsize = (6,4))
    # Defining low and high threshold values by their percentiles
    low_thresh = np.array([np.percentile(cv2lenna, 5),np.percentile(cv2lenna, 15),np.percentile(cv2lenna, 25)])
    high_thresh = np.array([np.percentile(cv2lenna, 65),np.percentile(cv2lenna, 80),np.percentile(cv2lenna, 95)])
    perc = np.array([5,15,25,65,80,95])
    # Histogram of pixel intensities
    plt.hist(flattened, alpha = 0.5, bins = 30)
    # Take every percentile value, and its percentile number
    for i, p in zip(np.concatenate((low_thresh, high_thresh), axis=None), perc):
        # Adding vertical line 
        ax.axvline(i, linestyle = "-", color = "Forestgreen") 
        # Adding text
        ax.text(i+2, # To create distance to vline
                1, # Y value
                s = f" {p}th percentile: {int(i)}", # Text
                rotation=90, # Rotation
                va='bottom') # Position of text

    plt.title("Percentiles")
    plt.savefig(os.path.join(output_path, "PixelHistogram.png"))


# Thresholds and scale
def threshold_canny(cv2lenna, output_path):
    # Constant scale value
    scale = 2
    # Smoothing using the Gaussian kernel
    smoothed = gaussian_filter(cv2lenna,
                            scale, # standard deviation for Gaussian kernel
                            order=0, # An order of 0 corresponds to convolution with a Gaussian kernel. A positive order corresponds to convolution with that derivative of a Gaussian.
                            output=None, # The array in which to place the output, or the dtype of the returned array. By default an array of the same dtype as input will be created.
                            mode='reflect', # ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
                            cval=0.0, # Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
                            truncate=4.0) # Truncate the filter at this many standard deviations. Default is 4.0.
    # Defining the thresholds in the hysterisis
    low_thresh = np.array([5,15,25])
    high_thresh = np.array([65, 80, 95])
    # Iterating through combinations of thresholds
    combi = []
    for i in low_thresh:
        for j in high_thresh:
            combi.append([i,j])
    # Figure properties
    fig, axes = plt.subplots(nrows=len(low_thresh), ncols=len(high_thresh), figsize=(10,10))
    ax_list = axes.flatten()
    # Hysterisis thresholding   
    for ax, thresholds in zip(ax_list, combi):
        canny = Canny(smoothed, np.percentile(cv2lenna, thresholds[0]), np.percentile(cv2lenna, thresholds[1]))
        ax.imshow(canny)
        ax.set_title(f"Threshold: lower = {thresholds[0]}, upper = {thresholds[1]}", fontsize = 10)

    plt.savefig(os.path.join(output_path, "ThresholdCanny.png"))

def final_canny(cv2lenna, output_path):
    scale = 2
    low = np.percentile(cv2lenna, 15)
    high = np.percentile(cv2lenna, 80)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    smoothed = gaussian_filter(cv2lenna,
                        scale, 
                        order=0,
                        output=None, 
                        cval=0.0,
                        truncate=4.0) 

    canny = Canny(smoothed, low, high) 
    axes[0].imshow(canny)

    # And on top of the original image
    # Defining the contours in the image
    (contours,_) = cv2.findContours(canny.copy(), # using np function to make a copy rather than destroying the image itself
                     cv2.RETR_EXTERNAL, 
                     cv2.CHAIN_APPROX_SIMPLE) 
    drawn = cv2.drawContours(
                     cv2lenna.copy(), # image, contours, fill, color, thickness
                     contours,
                     -1, # whihch contours to draw. -1 will draw contour for every contour that it finds
                     (255,0,0), # contour color
                     1) # Thickness
    axes[1].imshow(drawn)
    plt.savefig(os.path.join(output_path, "FinalCanny.png"))

    
    
# Defining the actions performed when running the file
def main(): # Now I'm defining the main function where I try to make it possible executing arguments from the terminal
    # add description
    ap = argparse.ArgumentParser(description = "[INFO] edge detection script argument") # Defining an argument parse

    ap.add_argument("-i", 
                "--input_path",  # Argument 1
                required=False, # Not required
                type = str, # Str input
                default = ".")
    ap.add_argument("-o", 
                "--output_path",  # Argument 2
                 required=False, # Not required
                 type = str, # Str input
                 default = ".")
    
    # Adding them together
    args = vars(ap.parse_args())
    
    lenna = imread(os.path.join(args['input_path'] , "lenna.jpg"))

    # Gaussian filtering - Show the result using scale = 1, 2, 4, 8 pixels.
    # Gradient magnitude computation using Gaussian derivatives - Use scale = 1, 2, 4, 8 pixels.
    gaussian_convolutions(lenna, 
                      gaus_matr, 
                      convolution,
                      args['output_path'])
    
    salt_pepper(lenna, args['output_path'])
    disk_gaussian(args['output_path'])
    

    # Laplacian-Gaussian filtering
    DoG(lenna, args['output_path']) 
    # save images with different barwidths to compare the effect of the difference of gaussian filter
    
    for i in range(1,7):
        bar_width = 2**i
        plots = plot_diff_gauss_bars(bar_width = bar_width)
        plots.savefig(os.path.join(args['output_path'], f'barwidth = {bar_width}'))
    
    DoG_shaded_bars(args['output_path'])
                      
    # Canny edge detection
    cv2lenna = cv2.imread(os.path.join(args['input_path'], "lenna.jpg"))
    scale_canny(cv2lenna, args['output_path'])
    pixel_histogram(cv2lenna, args['output_path'])
    threshold_canny(cv2lenna, args['output_path'])


if __name__ == "__main__":
                    main()
