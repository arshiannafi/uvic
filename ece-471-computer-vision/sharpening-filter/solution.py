import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap
from mpl_toolkits.mplot3d import Axes3D


# ---------------- Basic Image Handling ---------------------
# Input: image_path = string path to the image
# Output: HxWx1 numpy array containing the image
def read_image(image_path):
    # Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Input: image_to_save = 8 unsigned integer numpy array (image)
#        image_path = string path to save the image
# Output: True if image saved successfully, False otherwise
def save_image(image_to_save, image_path):
    # Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
    cv2.imwrite(image_path, image_to_save)


# Input: image_to_show = image to display
def show_image(image_to_show):
    # Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
    cv2.imshow('image', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise Exception('Even sized kernel')

    sigma2 = sigma ** 2
    offset = np.floor(size / 2)

    kernel = np.zeros((size, size))  # Init kernel

    for i in range(size):
        for j in range(size):
            # Source https://www.youtube.com/watch?v=LZRiMS0hcX4 (see details in report)
            x2_plus_y2 = (i - offset) ** 2 + (j - offset) ** 2

            kernel[i][j] = 1.0 * np.exp(-1.0 * x2_plus_y2 / (2.0 * sigma2)) / (2.0 * np.pi * sigma2)

    return kernel / np.sum(kernel)


# ---------------- Image Processing -------------------------
# Input: img = input HxWx1 grayscale image
#        ksize = size of the kernel (default=use 11x11 kernel)
#        padding = what kind of padding around the image do we want to use
# Output: return unsigned 8b integer image
def imfilter2d(img, ksize=9, padding=cv2.BORDER_REFLECT):
    # Create sharpen filter with a 9x9 Gaussian kernel with sigma 5, and unit
    #        impulse of 2
    kernel = gaussian_kernel(ksize, 5)

    # To find the location of the impulse in the kernel
    # and is used as offset in the convolution
    half_of_ksize = int(ksize / 2)
    # Converting a float to int drops the decimal. I.e. this also performs
    # the 'floor' operation

    # Unit impulse
    impulse = np.zeros((ksize, ksize))
    impulse[half_of_ksize, half_of_ksize] = 2

    # Source: Lecture 2 slides
    kernel = impulse - kernel

    # Create a float32 numpy array to save the result of the convolution
    result = np.zeros((img.shape[0], img.shape[1]), dtype='float32')

    # Apply border padding to the image
    img_padded = cv2.copyMakeBorder(img, half_of_ksize, half_of_ksize,
                                    half_of_ksize, half_of_ksize, padding)

    # Convolution. Source: http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html
    # Flipping the kernel before convolution.
    kernel = np.flipud(np.fliplr(kernel))

    # Sliding window
    for x in range(img_padded.shape[0] - ksize):
        for y in range(img_padded.shape[1] - ksize):
            # Get the neighbourhood
            image_local = img_padded[x:x + ksize, y:y + ksize]
            # Perform convolution
            resulting_pixel = (kernel * image_local).sum()
            # Store result
            result[x, y] = resulting_pixel

    # Clip the result so that the values are in the range (0,255) and save as unsigned 8 bit integer
    # Source https://stackoverflow.com/questions/3391843/how-to-transform-negative-elements-to-zero-without-a-loop
    result_clipped = result.clip(min=0, max=255)

    # Source: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    result_clipped = result_clipped.astype(np.uint8)

    return result_clipped


if __name__ == "__main__":
    # Handle arguments
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to image", default="pear.png")
    args = parser.parse_args()

    # Get the greyscale image
    img = read_image(args.image)
    # Show the image
    show_image(img)
    # Sharpen the image and return the result
    res = imfilter2d(img)
    # Show the sharpened image
    show_image(res)
    # Save the sharpened image as "sharpened.png"
    save_image(res, "sharpened.png")
