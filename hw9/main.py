import cv2
import math
import numpy as np


def apply_mask(img, y, x, mask):
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape
    r = 0
    for m_y in range(mask_height):
        for m_x in range(mask_width):
            if y + m_y < img_height and x + m_x < img_width:
                r += img[y + m_y, x + m_x] * mask[m_y, m_x]
    return r


def edge_detector1(img, mask1, mask2, threshold):
    img = img.copy()
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            r1 = apply_mask(img, y, x, mask1)
            r2 = apply_mask(img, y, x, mask2)
            g = math.sqrt(r1 ** 2 + r2 ** 2)
            if g >= threshold:
                img[y, x] = 0
            else:
                img[y, x] = 255

    return img


def roberts_operator(img):
    mask1 = np.array([[-1, 0], [0, 1]])
    mask2 = np.array([[0, -1], [1, 0]])
    return edge_detector1(img, mask1, mask2, 12)


def prewitts_edge_detector(img):
    mask1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    mask2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    return edge_detector1(img, mask1, mask2, 24)


def sobels_edge_detector(img):
    mask1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    mask2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return edge_detector1(img, mask1, mask2, 38)


def frei_n_chens_gradient_operator(img):
    mask1 = np.array([[-1, -math.sqrt(2), -1], [0, 0, 0], [1, math.sqrt(2), 1]])
    mask2 = np.array([[-1, 0, 1], [-math.sqrt(2), 0, math.sqrt(2)], [-1, 0, 1]])
    return edge_detector1(img, mask1, mask2, 30)


def edge_detector2(img, masks, threshold):
    img = img.copy()
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            all_g = []
            for mask in masks:
                g = apply_mask(img, y, x, mask)
                all_g.append(g)
            max_g = max(all_g)

            if max_g >= threshold:
                img[y, x] = 0
            else:
                img[y, x] = 255

    return img


def kirschs_compass_operator(img):
    masks = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
    ]
    return edge_detector2(img, masks, 125)


def robinsons_compass_operator(img):
    masks = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, -0]]),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
    ]
    return edge_detector2(img, masks, 43)


def nevatia_babu_5x5_operator(img):
    masks = [
        np.array(
            [
                [100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100],
                [0, 0, 0, 0, 0],
                [-100, -100, -100, -100, -100],
                [-100, -100, -100, -100, -100],
            ]
        ),
        np.array(
            [
                [100, 100, 100, 100, 100],
                [100, 100, 100, 78, -32],
                [100, 92, 0, -92, -100],
                [32, -78, -100, -100, -100],
                [-100, -100, -100, -100, -100],
            ]
        ),
        np.array(
            [
                [100, 100, 100, 32, -100],
                [100, 100, 92, -78, -100],
                [100, 100, 0, -100, -100],
                [100, 78, -92, -100, -100],
                [100, -32, -100, -100, -100],
            ]
        ),
        np.array(
            [
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, 0, 100, 100],
            ]
        ),
        np.array(
            [
                [-100, 32, 100, 100, 100],
                [-100, -78, 92, 100, 100],
                [-100, -100, 0, 100, 100],
                [-100, -100, -92, 78, 100],
                [-100, -100, -100, -32, 100],
            ]
        ),
        np.array(
            [
                [100, 100, 100, 100, 100],
                [-32, 78, 100, 100, 100],
                [-100, -92, 0, 92, 100],
                [-100, -100, -100, -78, 32],
                [-100, -100, -100, -100, -100],
            ]
        ),
    ]
    return edge_detector2(img, masks, 12500)


if __name__ == "__main__":
    original_img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    img = roberts_operator(original_img)
    cv2.imwrite("roberts_operator.jpg", img)

    img = prewitts_edge_detector(original_img)
    cv2.imwrite("prewitts_edge_detector.jpg", img)

    img = sobels_edge_detector(original_img)
    cv2.imwrite("sobels_edge_detector.jpg", img)

    img = frei_n_chens_gradient_operator(original_img)
    cv2.imwrite("frei_n_chens_gradient_operator.jpg", img)

    img = kirschs_compass_operator(original_img)
    cv2.imwrite("kirschs_compass_operator.jpg", img)

    img = robinsons_compass_operator(original_img)
    cv2.imwrite("robinsons_compass_operator.jpg", img)

    img = nevatia_babu_5x5_operator(original_img)
    cv2.imwrite("nevatia_babu_5x5_operator.jpg", img)
