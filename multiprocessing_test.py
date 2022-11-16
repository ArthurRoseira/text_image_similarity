import multiprocessing as mp
import time
import cv2
import numpy as np
from tqdm import tqdm

height = 512
width = 512
dict_images = {
    "img1": np.zeros((height, width, 3), np.uint8),
    "img2": np.ones((height, width, 3), np.uint8),
    "img3": np.full((height, width, 3), 2, np.uint8),
    "img4": np.full((height, width, 3), 3, np.uint8),
    "img5": np.full((height, width, 3), 4, np.uint8),
    "img6": np.full((height, width, 3), 5, np.uint8),
    "img7": np.full((height, width, 3), 6, np.uint8),
}

list_images = list(dict_images)
results= []

def compute_intensive_function(image_in):
    # Resizing
    image_in = dict_images[image_in]
    smaller_image = cv2.resize(image_in, (100, 100), interpolation=1)

    # Rotation
    rows, cols = image_in.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst_rotation = cv2.warpAffine(image_in, M, (cols, rows))

    # Translation
    M = np.float32([[1, 0, -100], [0, 1, -100]])
    dst_translation = cv2.warpAffine(image_in, M, (cols, rows))

    # Edge detection
    edges = cv2.Canny(image_in, 100, 200)

    # Kernel
    averaging_kernel = np.ones((3, 3), np.float32) / 9
    filtered_image = cv2.filter2D(image_in, -1, averaging_kernel)

    # Gaussian Kernel
    gaussian_kernel_x = cv2.getGaussianKernel(5, 1)
    gaussian_kernel_y = cv2.getGaussianKernel(5, 1)
    gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
    filtered_image = cv2.filter2D(image_in, -1, gaussian_kernel)

    # Image contours
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # calculate the contours from binary image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    with_contours = cv2.drawContours(image_in, contours, -1, (0, 255, 0), 3)
    return True


if __name__ == "__main__":

    list_images = list_images*1000

    with mp.Pool(16) as p:
        result = list(tqdm(p.imap(compute_intensive_function, list_images), total=7000))
        p.close()
        p.join()

    result = np.array(result)


