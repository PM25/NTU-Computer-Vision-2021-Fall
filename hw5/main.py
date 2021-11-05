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


if __name__ == "__main__":
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    kernel = get_octagon_kernel()

    dilation_img = dilation(img.copy(), kernel)
    cv2.imwrite(f"dilation.jpg", dilation_img)

    erosion_img = erosion(img.copy(), kernel)
    cv2.imwrite(f"erosion.jpg", erosion_img)

    opening_img = opening(img.copy(), kernel)
    cv2.imwrite(f"opening.jpg", opening_img)

    closing_img = closing(img.copy(), kernel)
    cv2.imwrite(f"closing.jpg", closing_img)
