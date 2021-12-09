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


def get_neighbors_pixel(img, y, x):
    height, width = img.shape

    coords = [
        (x, y),
        (x + 1, y),
        (x, y + 1),
        (x - 1, y),
        (x, y - 1),
        (x + 1, y - 1),
        (x + 1, y + 1),
        (x - 1, y + 1),
        (x - 1, y - 1),
    ]

    neighbors_pixel = []
    for x, y in coords:
        if x < width and x >= 0 and y < height and y >= 0:
            neighbors_pixel.append(img[y, x])

    return neighbors_pixel


def zero_crossing_edge_detection(img, mask, threshold):
    height, width = img.shape
    mask_img = img.copy().astype("int16")
    out_img = img.copy()

    m_height, m_width = mask.shape
    assert m_height == m_width and m_height % 2 == 1
    offset = m_height // 2
    pad_img = cv2.copyMakeBorder(
        img, offset, offset, offset, offset, cv2.BORDER_REFLECT
    )

    for y in range(height):
        for x in range(width):
            t = apply_mask(pad_img, y + offset, x + offset, mask)
            if t >= threshold:
                mask_img[y, x] = 1
            elif t <= -threshold:
                mask_img[y, x] = -1
            else:
                mask_img[y, x] = 0

    for y in range(height):
        for x in range(width):
            is_zero_crossing = False

            if mask_img[y, x] >= 1:
                for n in get_neighbors_pixel(mask_img, y, x):
                    if n <= -1:
                        out_img[y, x] = 0
                        is_zero_crossing = True
                        break

            if not is_zero_crossing:
                out_img[y, x] = 255

    return out_img


def laplace_mask1(img):
    mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return zero_crossing_edge_detection(img, mask, 15)


def laplace_mask2(img):
    mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) / 3
    return zero_crossing_edge_detection(img, mask, 15)


def minimum_variance_laplacian(img):
    mask = np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]) / 3
    return zero_crossing_edge_detection(img, mask, 20)


def laplace_of_gaussian(img):
    mask = np.array(
        [
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
        ]
    )
    return zero_crossing_edge_detection(img, mask, 3000)


def difference_of_gaussian(img):
    mask = np.array(
        [
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ]
    )
    return zero_crossing_edge_detection(img, mask, 1)


if __name__ == "__main__":
    original_img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    img = laplace_mask1(original_img)
    cv2.imwrite("laplace_mask1.jpg", img)

    img = laplace_mask2(original_img)
    cv2.imwrite("laplace_mask2.jpg", img)

    img = minimum_variance_laplacian(original_img)
    cv2.imwrite("minimum_variance_laplacian.jpg", img)

    img = laplace_of_gaussian(original_img)
    cv2.imwrite("laplace_of_gaussian.jpg", img)

    img = difference_of_gaussian(original_img)
    cv2.imwrite("difference_of_gaussian.jpg", img)
