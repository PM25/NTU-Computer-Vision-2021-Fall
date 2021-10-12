import cv2
import numpy as np


def up_down(img, fname="upside_down.bmp"):
    height, width, depth = img.shape

    for y in range(height // 2):
        rev_y = height - y - 1
        img[y, :, :], img[rev_y, :, :] = img[rev_y, :, :], img[y, :, :].copy()

    cv2.imwrite(fname, img)


def left_right(img, fname="left_right.bmp"):
    height, width, depth = img.shape

    for x in range(width // 2):
        rev_x = width - x - 1
        img[:, x, :], img[:, rev_x, :] = img[:, rev_x, :], img[:, x, :].copy()

    cv2.imwrite(fname, img)


def diagonally_flip(img, fname="diagonally_flip.bmp"):
    height, width, depth = img.shape

    for y in range(height):
        for x in range(y):
            img[y, x, :], img[x, y, :] = img[x, y, :], img[y, x, :].copy()

    cv2.imwrite(fname, img)


def rotate(img, degree=45, fname="rotate_45_degree.bmp"):
    height, width, depth = img.shape
    img_center = (height / 2, width / 2)

    rot_mat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    result = cv2.warpAffine(img, rot_mat, (height, width))

    cv2.imwrite(fname, result)


def shrink(img, ratio=0.5, fname="shrink_half.bmp"):
    height, weight, depth = img.shape
    dim = (int(height * ratio), int(weight * ratio))
    result = cv2.resize(img, dim, cv2.INTER_AREA)

    cv2.imwrite(fname, result)


def binarize(img, threshold=128, fname="binarize.bmp"):
    ret, result = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    cv2.imwrite(fname, result)


if __name__ == "__main__":
    img = cv2.imread("lena.bmp")

    # part 1
    up_down(img.copy())
    left_right(img.copy())
    diagonally_flip(img.copy())

    # part 2
    rotate(img.copy(), degree=-45)
    shrink(img.copy(), ratio=0.5)
    binarize(img.copy(), threshold=128)
