import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib

def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.sum(G)

def PSNR(f,g):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))

def plotGrayscaleImage(figure, title, image, rows=1, columns=1, position=1):
    figure.add_subplot(rows, columns, position)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title)

def imnoise_speckle(im, v): # im: input image
     # v:  variance
     n = np.sqrt(v*12) * (np.random.rand(im.shape[0], im.shape[1]) - 0.5)
     return im + im * n

def plot_convolved_images(original_image, image_title):
    h1 = (1/6)*np.ones((1,6))
    h2 = h1.T
    h3 = np.array([[-1,1]])

    original_image_h1 = signal.convolve2d(original_image, h1)
    original_image_h2 = signal.convolve2d(original_image, h2)
    original_image_h3 = signal.convolve2d(original_image, h3)

    fig = plt.figure(figsize=(15, 10))
    rows = 2
    columns = 2

    plotGrayscaleImage(fig, 'Original', original_image, rows, columns, 1)
    plotGrayscaleImage(fig, f'{image_title} * h1', original_image_h1, rows, columns, 2)
    plotGrayscaleImage(fig, f'{image_title} * h2', original_image_h2, rows, columns, 3)
    plotGrayscaleImage(fig, f'{image_title} * h3', original_image_h3, rows, columns, 4)
    plt.show()

def plot_images_with_noise():
    f = np.hstack([0.3*np.ones((200,100)), 0.7*np.ones((200,100))])
    f_gaussian_noise = skimage.util.random_noise(f, mode='gaussian', var=0.01)
    f_salt_pepper_noise = skimage.util.random_noise(f, mode='s&p', amount=0.05)
    f_speckle_noise = imnoise_speckle(f, 0.04)

    fig = plt.figure(figsize=(15, 15))
    rows = 2
    columns = 2

    # Plot noise images
    plotGrayscaleImage(fig, 'Original', f, rows, columns, 1)
    plotGrayscaleImage(fig, 'Gaussian Noise', f_gaussian_noise, rows, columns, 2)
    plotGrayscaleImage(fig, 'Salt and Pepper Noise', f_salt_pepper_noise, rows, columns, 3)
    plotGrayscaleImage(fig, 'Speckle Noise', f_speckle_noise, rows, columns, 4)
    plt.show()

    fig = plt.figure(figsize=(15, 15))
    rows = 2
    columns = 2
    
    # Histograms of noise images
    plot_hist(fig, f, 'Histogram of Original', rows, columns, 1)
    plot_hist(fig, f_gaussian_noise, 'Histogram of Gaussian Noise', rows, columns, 2)
    plot_hist(fig, f_salt_pepper_noise, 'Histogram of Salt and Pepper Noise', rows, columns, 3)
    plot_hist(fig, f_speckle_noise, 'Histogram of Speckle Noise', rows, columns, 4)
    plt.show()

def plot_hist(figure, image, title, rows, columns, position):
    figure.add_subplot(rows, columns, position)
    plt.hist(image.flatten())
    plt.title(title)

def main():
    plt.gray()
    lena= rgb2gray(imread('lena.tiff'))
    cameraman = imread('cameraman.tif').astype(np.float64)/255

    # 2: Discrete Convolution for Image Processing
    plot_convolved_images(lena, 'Lena')

    # 3: Noise Generation
    plot_images_with_noise()
    figure = plt.figure(figsize=(10, 10))
    plotGrayscaleImage(figure, 'Original Lena', lena, 1, 2, 1)
    plot_hist(figure, lena, 'Histogram of Original Lena', 1, 2, 2)
    plt.show()

    # 4: Noise Reduction in the Spatial Domain
    small_averaging_filter = np.ones((3,3))/(3.0*3.0)
    # Convole with small averaging filter
    lena_small_averaging_filter = ndimage.convolve(lena, small_averaging_filter)

    # Plot averaging filter
    # Load the Lena image and convert it to a grayscale image using the rgb2gray
    # function. To evaluate the noise reduction performance of various noise reduction
    # techniques, we will contaminate the Lena image with zero-mean Gaussian
    # noise with a variance of 0.002. Plot the noisy image and the corresponding 
    # histogram and PSNR between the noisy image and the original noise-free
    # image.
    lena_gaussian_noise = skimage.util.random_noise(lena, mode='gaussian', var=0.002)

    figure = plt.figure(figsize=(10, 5))
    plotGrayscaleImage(figure, 'Lena Gaussian Noise', lena_gaussian_noise, 1, 2, 1)
    plot_hist(figure, lena_gaussian_noise, 'Histogram of Lena Gaussian Noise', 1, 2, 2)
    plt.show()
    print(f'PSNR of Lena Gaussian Noise: {PSNR(lena, lena_gaussian_noise)}')

    plt.imshow(lena_small_averaging_filter)
    plt.title("Original Lena with small averaging filter")
    plt.show()

    # Now apply the averaging filter to the noisy image using the ndimage.convolve 
    # function. Plot the denoised image and the corresponding histogram. Also, compute 
    # the PSNR between the denoised image and the original noise-free image.
    lena_denoised_averaging = ndimage.convolve(lena_gaussian_noise, small_averaging_filter)
    figure = plt.figure(figsize=(10, 10))
    plotGrayscaleImage(figure, 'Lena Gaussian Noise', lena_gaussian_noise, 2, 2, 1)
    plotGrayscaleImage(figure, 'Lena Denoised Small Filter', lena_denoised_averaging, 2, 2, 2)
    plot_hist(figure, lena_gaussian_noise, 'Histogram of Lena Gaussian Noise', 2, 2, 3)
    plot_hist(figure, lena_denoised_averaging, 'Histogram of Lena Denoised with Small Filter', 2, 2, 4)
    plt.show()
    print(f'PSNR of Lena Denoised with Small Filter: {PSNR(lena, lena_denoised_averaging)}')

    # Let us now create a 7×7 averaging filter kernel and apply it to the noisy image. 
    # Plot the denoised image and the corresponding histogram. Also, compute the PSNR 
    # between the denoised image and the original noise-free image.
    figure = plt.figure(figsize=(10, 5))
    large_averaging_filter = np.ones((7,7))/(7.0*7.0)
    lena_denoised_large_averaging = ndimage.convolve(lena_gaussian_noise, large_averaging_filter)
    plotGrayscaleImage(figure, 'Lena Denoised with Large Filter', lena_denoised_large_averaging, 1, 2, 1)
    plot_hist(figure, lena_denoised_large_averaging, 'Histogram of Lena Denoised with Large Filter', 1, 2, 2)
    plt.show()
    print(f'PSNR of Lena Denoised with Large Filter: {PSNR(lena, lena_denoised_large_averaging)}')
     
    #  Let us now create a 7×7 Gaussian filter kernel with a standard deviation of 1 using the provided gaussian filter function. 
    # Plot the filter. Plot the denoised image and the corresponding histogram. Also, compute the PSNR between the denoised image and the original noise-free image.
    lena_denoised_gaussian = ndimage.gaussian_filter(lena_gaussian_noise, sigma=1)
    figure = plt.figure(figsize=(10, 5))
    plotGrayscaleImage(figure, 'Lena denoised with Gaussian Filter', lena_denoised_gaussian, 1, 2, 1)
    plot_hist(figure, lena_denoised_gaussian, 'Histogram of Lena Denoised with Gaussian Filter', 1, 2, 2)
    plt.show()
    print(f'PSNR of Lena Denoised with Gaussian Filter: {PSNR(lena, lena_denoised_gaussian)}')

    # Let us now create a new noisy image by adding salt and pepper noise (density 0.05) to the image. Apply the 7×7 averaging 
    # filter and the Gaussian filter to the noisy image separately. Plot the noisy image, the denoised images using each method, 
    # and the corresponding histograms. Also, compute the PSNR between the denoised images and the original noise-free image.
    lena_salt_pepper_noise = skimage.util.random_noise(lena, mode='s&p', amount=0.05)
    figure = plt.figure(figsize=(10,10))
    plotGrayscaleImage(figure, 'Lena Salt and Pepper Noise', lena_salt_pepper_noise, 1, 2, 1)
    plot_hist(figure, lena_salt_pepper_noise, 'Histogram of Lena Salt and Pepper Noise', 1, 2, 2)
    plt.show()

    lena_salt_pepper_denoised_gaussian = ndimage.gaussian_filter(lena_salt_pepper_noise, sigma=1)
    lena_salt_pepper_denoised_averaging = ndimage.convolve(lena_salt_pepper_noise, large_averaging_filter)
    figure = plt.figure(figsize=(10,10))
    plotGrayscaleImage(figure, 'Lena Salt and Pepper Denoised with Gaussian Filter', lena_salt_pepper_denoised_gaussian, 2, 2, 1)
    plotGrayscaleImage(figure, 'Lena Salt and Pepper Denoised with Average Filter', lena_salt_pepper_denoised_averaging, 2, 2, 2)
    plot_hist(figure, lena_salt_pepper_denoised_gaussian, 'Histogram of Lena Denoised with Gaussian Filter',2, 2, 3)
    plot_hist(figure, lena_denoised_gaussian, 'Histogram of Lena Denoised with Average Filter', 2, 2, 4)
    plt.show()

    print(f'PSNR of Salt and Pepper Lena: {PSNR(lena, lena_salt_pepper_noise)}')
    print(f'PSNR of Salt and Pepper Lena Denoised with Gaussian Filter: {PSNR(lena, lena_salt_pepper_denoised_gaussian)}')
    print(f'PSNR of Salt and Pepper Lena Denoised with Average Filter: {PSNR(lena, lena_salt_pepper_denoised_averaging)}')

    # Let us now apply a 3x3 median filter on the noisy image. The ndimage.median filter function will come in handy for this.
    # Plot the denoised image and the corresponding histogram. Also, compute the PSNR between the denoised image and the original noise-free image.
    lena_denoised_median = ndimage.median_filter(lena_gaussian_noise, size=3)
    lena_salt_pepper_denoised_median = ndimage.median_filter(lena_salt_pepper_noise, size=3)
    figure = plt.figure(figsize=(10,10))
    plotGrayscaleImage(figure, 'Lena Gaussian Denoised with Median Filter', lena_denoised_median, 2, 2, 1)
    plotGrayscaleImage(figure, 'Lena Salt and Pepper Denoised with Median Filter', lena_salt_pepper_denoised_median, 2, 2, 2)
    plot_hist(figure, lena_denoised_median, 'Histogram of Lena Denoised with Median Filter',2, 2, 3)
    plot_hist(figure, lena_salt_pepper_denoised_median, 'Histogram of Salt and Pepper Lena Denoised with Median Filter', 2, 2, 4)
    plt.show()

    print(f'PSNR of Gaussian Noise Lena Denoised with Median Filter: {PSNR(lena, lena_denoised_median)}')
    print(f'PSNR of Salt and Pepper Lena Denoised with Median Filter: {PSNR(lena, lena_salt_pepper_denoised_median)}')

    # 5: Sharpening in the Spatial Domain
    figure = plt.figure(figsize=(15,10))
    plot_hist(figure, cameraman, "Histogram of cameraman images", 2, 3, 1)
    cameraman_gaussian = ndimage.gaussian_filter(cameraman, sigma=1)
    cameraman_minus_gaussian = np.subtract(cameraman, cameraman_gaussian)
    plotGrayscaleImage(figure, "Cameraman with Gaussian Filter", cameraman_gaussian, 2, 3, 2)
    plotGrayscaleImage(figure, "Cameraman Subtracted by Gaussian Filtered Image", cameraman_minus_gaussian, 2, 3, 3)
    plotGrayscaleImage(figure, "Original Cameraman", cameraman, 2, 3, 4)

    cameraman_plus_subtracted_image = np.add(cameraman, cameraman_minus_gaussian)
    plotGrayscaleImage(figure, "Cameraman Plus Subtracted Image", cameraman_plus_subtracted_image, 2, 3, 5)

    # Now, instead of adding the subtracted image to the original image, multiply the 
    # subtracted image by 0.5 and then add it to the original image. Plot the resulting image.
    cameraman_plus_scaled_image = np.add(cameraman, np.multiply(cameraman_minus_gaussian,0.5))
    plotGrayscaleImage(figure, "Cameraman Plus Scaled Subtracted Image", cameraman_plus_scaled_image, 2, 3, 6)
    plt.show()

    figure = plt.figure(figsize=(15,15))
    plot_hist(figure, lena, 'Histogram of Original Lena', 3, 3, 1)
    plot_hist(figure, lena_gaussian_noise, 'Histogram of Lena Gaussian Noise', 3, 3, 2)
    plot_hist(figure, lena_denoised_averaging, 'Histogram of Lena Denoised with Small Filter', 3, 3, 3)
    plot_hist(figure, lena_denoised_large_averaging, 'Histogram of Lena Denoised with Large Filter', 3, 3, 4)
    plot_hist(figure, lena_denoised_gaussian, 'Histogram of Lena Denoised with Gaussian Filter', 3, 3, 5)
    plot_hist(figure, lena_salt_pepper_noise, 'Histogram of Salt and Pepper Lena', 3, 3, 6)
    plot_hist(figure, lena_denoised_median, 'Histogram of Gaussian Lena denoised with Median Filter',3, 3, 7)
    plot_hist(figure, lena_salt_pepper_denoised_median, 'Histogram of S&P Lena Denoised with Median Filter', 3, 3, 8)
    
    plt.show()





    


if __name__ == '__main__':
    main()
