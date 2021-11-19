import cv2
import numpy as np


def binarize(img):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            if img[y, x] < 128:
                img[y, x] = 0
            else:
                img[y, x] = 255

    return img


def downsample(img, kernel_sz=8):
    height, width = img.shape
    out_img = np.zeros((height // kernel_sz, width // kernel_sz))

    for y in range(0, height, kernel_sz):
        for x in range(0, width, kernel_sz):
            out_img[y // kernel_sz, x // kernel_sz] = img[y, x]

    return out_img


def h(b, c, d, e):
    if b == c and (b != d or b != e):
        return "q"
    elif b == c == d == e:
        return "r"
    elif b != c:
        return "s"


def f(a1, a2, a3, a4):
    if a1 == a2 == a3 == a4 == "r":
        return 5

    return len([a for a in [a1, a2, a3, a4] if a == "q"])


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
        else:
            neighbors_pixel.append(0)

    return neighbors_pixel


def yokoi_connectivity(img):
    height, width = img.shape
    out_img = img.copy()

    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                x_i = get_neighbors_pixel(img, y, x)
                a1 = h(x_i[0], x_i[1], x_i[6], x_i[2])
                a2 = h(x_i[0], x_i[2], x_i[7], x_i[3])
                a3 = h(x_i[0], x_i[3], x_i[8], x_i[4])
                a4 = h(x_i[0], x_i[4], x_i[5], x_i[1])
                n = f(a1, a2, a3, a4)
                out_img[y, x] = n
            else:
                out_img[y, x] = 0

    return out_img


def pair_relationship_operator(img):
    height, width = img.shape
    out_img = img.copy()
    out_img = out_img.astype("str")

    for y in range(height):
        for x in range(width):
            if img[y, x] == 1:
                is_p = False
                x_is = get_neighbors_pixel(img, y, x)
                for x_i in x_is[1:5]:  # x_1, x_2, x_3, x_4
                    if x_i is not None and x_i == 1:
                        is_p = True

                out_img[y, x] = "p" if is_p else "q"

            elif img[y, x] > 1:
                out_img[y, x] = "q"

            else:
                out_img[y, x] = " "

    return out_img


def connected_shrink_operator(original_img, marked_img):
    height, width = marked_img.shape
    out_img = original_img.copy()

    for y in range(height):
        for x in range(width):
            if marked_img[y, x] == "p":
                x_i = get_neighbors_pixel(out_img, y, x)
                a1 = h(x_i[0], x_i[1], x_i[6], x_i[2])
                a2 = h(x_i[0], x_i[2], x_i[7], x_i[3])
                a3 = h(x_i[0], x_i[3], x_i[8], x_i[4])
                a4 = h(x_i[0], x_i[4], x_i[5], x_i[1])

                n = f(a1, a2, a3, a4)
                if n == 1:
                    out_img[y, x] = 0

    return out_img


def thinning_operator(img):
    while True:
        yokoi_img = yokoi_connectivity(img)
        marked_img = pair_relationship_operator(yokoi_img)
        thinning_img = connected_shrink_operator(img, marked_img)
        if np.array_equal(img, thinning_img):
            break
        img = thinning_img

    return thinning_img


if __name__ == "__main__":
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    binarized_img = binarize(img)
    downsampled_img = downsample(binarized_img)
    thinning_img = thinning_operator(downsampled_img)
    cv2.imwrite(f"thinning.jpg", thinning_img)

