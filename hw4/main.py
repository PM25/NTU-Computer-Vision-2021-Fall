import cv2


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


def get_hit_and_miss_kernel():
    j = [(0, -1), (0, 0), (1, 0)]
    k = [(-1, 0), (-1, 1), (0, 1)]
    return j, k


def binarize(img):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            if img[y, x] < 128:
                img[y, x] = 0
            else:
                img[y, x] = 255

    return img


def dilation(img, kernel):
    height, width = img.shape

    out_img = img.copy()
    out_img[:, :] = 0

    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                for rel_y, rel_x in kernel:
                    new_y = y + rel_y
                    new_x = x + rel_x
                    if new_y < height and new_y >= 0 and new_x < width and new_x >= 0:
                        out_img[new_y, new_x] = 255

    return out_img


def erosion(img, kernel):
    height, width = img.shape

    out_img = img.copy()
    out_img[:, :] = 0

    if (0, 0) in kernel:
        target_pixel = 255
    else:
        target_pixel = 0

    for y in range(height):
        for x in range(width):
            if img[y, x] == target_pixel:
                check = True
                for rel_y, rel_x in kernel:
                    check_y = y + rel_y
                    check_x = x + rel_x
                    if (
                        check_y < height
                        and check_y >= 0
                        and check_x < width
                        and check_x >= 0
                        and img[check_y, check_x] == 0
                    ):
                        check = False
                        break

                if check:
                    out_img[y, x] = 255

    return out_img


def opening(img, kernel):
    erosion_img = erosion(img, kernel)
    out_img = dilation(erosion_img, kernel)
    return out_img


def closing(img, kernel):
    dilation_img = dilation(img, kernel)
    out_img = erosion(dilation_img, kernel)
    return out_img


def hit_and_miss(img, kernel_j, kernel_k):
    complement_img = img.copy()
    complement_img[img == 255] = 0
    complement_img[img == 0] = 255

    erosion_a_j = erosion(img.copy(), kernel_j)
    erosion_ac_k = erosion(complement_img.copy(), kernel_k)

    out_img = img.copy()
    out_img[:, :] = 0

    height, width = img.shape
    for y in range(height):
        for x in range(width):
            if erosion_a_j[y, x] == erosion_ac_k[y, x] == 255:
                out_img[y, x] = 255

    return out_img


if __name__ == "__main__":
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    binarize_img = binarize(img)
    kernel = get_octagon_kernel()

    dilation_img = dilation(binarize_img.copy(), kernel)
    cv2.imwrite(f"dilation.jpg", dilation_img)

    erosion_img = erosion(binarize_img.copy(), kernel)
    cv2.imwrite(f"erosion.jpg", erosion_img)

    opening_img = opening(binarize_img.copy(), kernel)
    cv2.imwrite(f"opening.jpg", opening_img)

    closing_img = closing(binarize_img.copy(), kernel)
    cv2.imwrite(f"closing.jpg", closing_img)

    kernel_j, kernel_k = get_hit_and_miss_kernel()

    hit_and_miss_img = hit_and_miss(binarize_img.copy(), kernel_j, kernel_k)
    cv2.imwrite(f"hit_and_miss.jpg", hit_and_miss_img)
