import cv2
import matplotlib.pyplot as plt


def histogram(img):
    height, width = img.shape

    pixel_count = [0] * 256
    for y in range(height):
        for x in range(width):
            pixel_count[img[y, x]] += 1

    return pixel_count


def divide_intensity(img, value=3):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            img[y, x] //= value

    return img


def histogram_equalization(img):
    height, width = img.shape
    hist = histogram(img)
    total_pixels = height * width
    cdf = 0

    s_k = []
    for i in range(256):
        cdf += hist[i]
        s_k.append(cdf * 255 / total_pixels)

    for y in range(height):
        for x in range(width):
            img[y, x] = s_k[img[y, x]]

    return img


def save(img, hist, fname):
    plt.bar(range(256), hist, 1)
    plt.savefig(f"{fname}_histogram.jpg")
    plt.clf()
    cv2.imwrite(f"{fname}.jpg", img)


if __name__ == "__main__":
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    hist = histogram(img.copy())
    save(img, hist, "original_image")

    divided_img = divide_intensity(img.copy(), 3)
    divided_hist = histogram(divided_img.copy())
    save(divided_img, divided_hist, "divided_image")

    equalized_img = histogram_equalization(divided_img.copy())
    equalized_hist = histogram(equalized_img.copy())
    save(equalized_img, equalized_hist, "equalized_image")
