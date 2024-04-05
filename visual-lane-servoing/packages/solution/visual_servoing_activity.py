from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    # Initialize the matrix with zeros
    steer_matrix_right = np.zeros(shape)
    
    # Fill the matrix with values that linearly increase from right to left
    for i in range(shape[1]):
        steer_matrix_right[:, i] = i / shape[1]
    
    # # Optionally, you can invert the direction if needed, so that the leftmost edge has the highest value
    # steer_matrix_right = np.fliplr(steer_matrix_right)
    
    return steer_matrix_right


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """
    # Initialize the matrix with zeros
    steer_matrix_left = np.zeros(shape)
    
    # Fill the matrix with values that linearly increase from right to left
    for i in range(shape[1]):
        steer_matrix_left[:, i] = i / shape[1]
    
    # # Optionally, you can invert the direction if needed, so that the leftmost edge has the highest value
    steer_matrix_left = np.fliplr(steer_matrix_left)
    
    return steer_matrix_left


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    threshold = 85 # CHANGE ME
    sigma = 1.5 # CHANGE ME

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    mask_mag = (Gmag > threshold)
    # white_lower_hsv = np.array([0 / 359 * 179, 2 / 100 * 255, 48 / 100 * 255])          # CHANGE ME
    # white_upper_hsv = np.array([353.3 / 359 * 179, 17 / 100 * 255, 110 / 100 * 255])    # CHANGE ME
    # yellow_lower_hsv = np.array([46 / 359 * 179, 45 / 100 * 255, 51  / 100 * 255])    # CHANGE ME
    # yellow_upper_hsv = np.array([56 / 359 * 179, 100 / 100 * 255, 93  / 100 * 255])   # CHANGE ME
    white_lower_hsv = np.array([0 / 360 * 179, 2 / 100 * 255, 55 / 100 * 255])          # CHANGE ME
    white_upper_hsv = np.array([360 / 360 * 179, 22 / 100 * 255, 100 / 100 * 255])    # CHANGE ME
    yellow_lower_hsv = np.array([41 / 360 * 179, 20 / 100 * 255, 40  / 100 * 255])    # CHANGE ME
    yellow_upper_hsv = np.array([58 / 360 * 179, 100 / 100 * 255, 100  / 100 * 255])   # CHANGE ME


    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    # TODO: implement your own solution here
    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
