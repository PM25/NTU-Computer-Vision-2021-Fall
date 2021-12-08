import cv2
import math
import random
import numpy as np

random.seed(1009)

box3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
box5x5 = np.array([[1] * 5] * 5)


def get_gaussian_noise_img(img, amp=10):
    img = img.copy()
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            img[y, x] += amp * random.gauss(0, 1)
            if img[y, x] > 255:
                img[y, x] = 255
    return img


def get_salt_n_pepper(img, threshold=0.05):
    img = img.copy()
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            random_value = random.uniform(0, 1)
            if random_value <= threshold:
                img[y, x] = 0
            elif random_value >= 1 - threshold:
                img[y, x] = 255

    return img


def box_filter(img, y, x, box):
    b_height, b_width = box.shape
    assert b_height == b_width and b_height % 2 == 1
    offset = b_height // 2

    s = 0
    for b_y in range(b_height):
        for b_x in range(b_width):
            s += img[y + b_y - offset, x + b_x - offset] * box[b_y, b_x]

    return s / np.sum(box)


def median_filter(img, y, x, box):
    b_height, b_width = box.shape
    assert b_height == b_width and b_height % 2 == 1
    offset = b_height // 2

    s = []
    for b_y in range(b_height):
        for b_x in range(b_width):
            s.append(img[y + b_y - offset, x + b_x - offset] * box[b_y, b_x])
    s.sort()

    return s[len(s) // 2]


def apply_filter(img, box, _filter="box"):
    img = img.copy()
    height, width = img.shape

    b_height, b_width = box.shape
    assert b_height == b_width and b_height % 2 == 1
    offset = b_height // 2
    pad_img = cv2.copyMakeBorder(
        img, offset, offset, offset, offset, cv2.BORDER_REFLECT
    )

    if _filter == "box":
        _filter = box_filter
    elif _filter == "median":
        _filter = median_filter

    for y in range(height):
        for x in range(width):
            img[y, x] = _filter(pad_img, y + offset, x + offset, box)

    return img


def get_octagon_kernel():
    octagon = []

    for y in range(-2, 3):
        for x in range(-2, 3):
            octagon.append((y, x))

    octagon.remove((-2, -2))
    octagon.remove((-2, 2))
    octagon.remove((2, -2))
    octagon.remove((2, 2))

    return octagon


def dilation(img, kernel):
    height, width = img.shape

    out_img = img.copy()
    out_img[:, :] = 0

    for y in range(height):
        for x in range(width):
            max_value = 0
            for rel_y, rel_x in kernel:
                new_y = y + rel_y
                new_x = x + rel_x
                if new_y < height and new_y >= 0 and new_x < width and new_x >= 0:
                    if img[new_y, new_x] > max_value:
                        max_value = img[new_y, new_x]

            out_img[y, x] = max_value

    return out_img


def erosion(img, kernel):
    height, width = img.shape

    out_img = img.copy()
    out_img[:, :] = 0

    for y in range(height):
        for x in range(width):
            min_value = 255
            for rel_y, rel_x in kernel:
                new_y = y + rel_y
                new_x = x + rel_x
                if new_y < height and new_y >= 0 and new_x < width and new_x >= 0:
                    if img[new_y, new_x] < min_value:
                        min_value = img[new_y, new_x]

            out_img[y, x] = min_value

    return out_img


def opening(img, kernel):
    erosion_img = erosion(img, kernel)
    out_img = dilation(erosion_img, kernel)
    return out_img


def closing(img, kernel):
    dilation_img = dilation(img, kernel)
    out_img = erosion(dilation_img, kernel)
    return out_img


def opening_closing(img, kernel):
    out_img = opening(img, kernel)
    out_img = closing(out_img, kernel)
    return out_img


def closing_opening(img, kernel):
    out_img = closing(img, kernel)
    out_img = opening(out_img, kernel)
    return out_img


def calc_signal_to_noise(original_img, noise_img):
    assert original_img.shape == noise_img.shape
    height, width = original_img.shape
    n = height * width

    # normalize
    original_img = original_img.copy().astype(float)
    original_img /= 255.0

    noise_img = noise_img.copy().astype(float)
    noise_img /= 255.0

    us = 0
    for y in range(height):
        for x in range(width):
            us += original_img[y, x]
    us /= n

    u_noise = 0
    for y in range(height):
        for x in range(width):
            u_noise += noise_img[y, x] - original_img[y, x]
    u_noise /= n

    vs, vn = 0, 0
    for y in range(height):
        for x in range(width):
            vs += (original_img[y, x] - us) ** 2
            vn += (noise_img[y, x] - original_img[y, x] - u_noise) ** 2
    vs /= n
    vn /= n

    snr = 20 * math.log10(math.sqrt(vs) / math.sqrt(vn))
    return snr


def save_img(original_img, noise_img, fname):
    snr = calc_signal_to_noise(original_img, noise_img)
    cv2.imwrite(f"{fname}_SNR_{snr:.5f}.jpg", noise_img)


if __name__ == "__main__":
    original_img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    kernel = get_octagon_kernel()

    # assignment a
    gaussian10_noise_img = get_gaussian_noise_img(original_img, amp=10)
    save_img(original_img, gaussian10_noise_img, "gaussian10_noise")
    gaussian30_noise_img = get_gaussian_noise_img(original_img, amp=30)
    save_img(original_img, gaussian30_noise_img, "gaussian30_noise")

    # assignment b
    salt_n_pepper_dot1_img = get_salt_n_pepper(original_img, threshold=0.1)
    save_img(original_img, salt_n_pepper_dot1_img, "salt_and_pepper_dot1")
    salt_n_pepper_dot05_img = get_salt_n_pepper(original_img, threshold=0.05)
    save_img(original_img, salt_n_pepper_dot05_img, "salt_and_pepper_dot05")

    noise_imgs = [
        gaussian10_noise_img,
        gaussian30_noise_img,
        salt_n_pepper_dot1_img,
        salt_n_pepper_dot05_img,
    ]
    out_fnames = [
        "gaussian10_noise",
        "gaussian30_noise",
        "salt_n_pepper_dot1",
        "salt_n_pepper_dot05",
    ]

    # assignment c
    box_filter_imgs = map(lambda img: apply_filter(img, box3x3, "box"), noise_imgs)
    for box_filter_img, fname in zip(box_filter_imgs, out_fnames):
        save_img(original_img, box_filter_img, f"{fname}_box3x3")

    box_filter_imgs = map(lambda img: apply_filter(img, box5x5, "box"), noise_imgs)
    for box_filter_img, fname in zip(box_filter_imgs, out_fnames):
        save_img(original_img, box_filter_img, f"{fname}_box5x5")

    # assignment d
    median_filter_imgs = map(
        lambda img: apply_filter(img, box3x3, "median"), noise_imgs
    )
    for median_filter_img, fname in zip(median_filter_imgs, out_fnames):
        save_img(original_img, median_filter_img, f"{fname}_median3x3")

    median_filter_imgs = map(
        lambda img: apply_filter(img, box5x5, "median"), noise_imgs
    )
    for median_filter_img, fname in zip(median_filter_imgs, out_fnames):
        save_img(original_img, median_filter_img, f"{fname}_median5x5")

    # assignment e
    opening_closing_imgs = map(lambda img: opening_closing(img, kernel), noise_imgs)
    for opening_closing_img, fname in zip(opening_closing_imgs, out_fnames):
        save_img(original_img, opening_closing_img, f"{fname}_opening_closing")

    closing_opening_imgs = map(lambda img: closing_opening(img, kernel), noise_imgs)
    for closing_opening_img, fname in zip(closing_opening_imgs, out_fnames):
        save_img(original_img, closing_opening_img, f"{fname}_closing_opening")
