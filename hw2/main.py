import cv2
import matplotlib.pyplot as plt


def binarize(img):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            if img[y, x] < 128:
                img[y, x] = 0
            else:
                img[y, x] = 255

    cv2.imwrite("binarize.jpg", img)
    return img


def histogram(img):
    height, width = img.shape

    pixel_count = [0] * 256
    for y in range(height):
        for x in range(width):
            pixel_count[img[y, x]] += 1

    plt.bar(range(256), pixel_count, 1)
    plt.savefig("histogram.jpg")


def find_neighbors(labels, center_y, center_x):
    neighbors = set()

    for y in range(center_y - 1, center_y + 2):
        for x in range(center_x - 1, center_x + 2):
            if (y, x) in labels:
                neighbors.add(labels[(y, x)])

    return neighbors


def get_centroid(positions):
    centroid_x, centroid_y = 0, 0

    for (y, x) in positions:
        centroid_x += x
        centroid_y += y

    return (centroid_y / len(positions), centroid_x / len(positions))


def draw_bounding_box(img, positions, color=(0, 255, 0)):
    ys = [y for (y, x) in positions]
    xs = [x for (y, x) in positions]
    cv2.rectangle(img, (min(xs), min(ys)), (max(xs), max(ys)), color, 2)


def draw_cross(img, y, x, color=(0, 0, 255)):
    cv2.line(img, (x, y - 10), (x, y + 10), color, 2)
    cv2.line(img, (x - 10, y), (x + 10, y), color, 2)


def union_find(table, x):
    if x not in table or table[x] == x:
        return x
    else:
        return union_find(table, table[x])


# class UnionFind:
#     def find(self, x):
#         pass

#     def union(self, x, y):
#         pass


def init_labels(img):
    height, width = img.shape
    labels = {}
    idx = 0

    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                labels[(y, x)] = idx
                idx += 1

    return labels


def connected_components_classical(img):
    height, width = img.shape
    labels = init_labels(img)

    # first top-down
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                labels[(y, x)] = min(find_neighbors(labels, y, x))

    table = {}
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                neighbors = find_neighbors(labels, y, x)
                if len(neighbors) > 1:
                    min_label = min(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in table:
                            table[neighbor] = min_label
                        elif table[neighbor] > min_label:
                            table[neighbor] = min_label

    for label in table:
        table[label] = union_find(table, label)

    reverse_labels = {}
    for pos, label in labels.items():
        label = union_find(table, label)

        if label in reverse_labels:
            reverse_labels[label].append(pos)
        else:
            reverse_labels[label] = [pos]

    color_img = cv2.merge([img, img, img])

    for label, positions in reverse_labels.items():
        if len(positions) >= 500:
            y, x = get_centroid(positions)
            draw_cross(color_img, int(y), int(x))
            draw_bounding_box(color_img, positions)

    cv2.imwrite("connected_components.jpg", color_img)


def connected_components_iterative(img, threshold=500):
    height, width = img.shape
    pos2label = init_labels(img)

    change = True
    while change:
        change = False
        # top-down
        for y in range(height):
            for x in range(width):
                if img[y, x] == 255:
                    min_label = min(find_neighbors(pos2label, y, x))
                    if pos2label[(y, x)] != min_label:
                        pos2label[(y, x)] = min_label
                        change = True

        # bottom-up
        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                if img[y, x] == 255:
                    min_label = min(find_neighbors(pos2label, y, x))
                    if pos2label[(y, x)] != min_label:
                        pos2label[(y, x)] = min_label
                        change = True

    label2pos = {}
    for pos, label in pos2label.items():
        if label in label2pos:
            label2pos[label].append(pos)
        else:
            label2pos[label] = [pos]

    color_img = cv2.merge([img, img, img])

    for label, positions in label2pos.items():
        if len(positions) >= threshold:
            y, x = get_centroid(positions)
            draw_cross(color_img, int(y), int(x))
            draw_bounding_box(color_img, positions)

    cv2.imwrite("connected_components.jpg", color_img)


if __name__ == "__main__":
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    binarize_img = binarize(img.copy())
    histogram(img.copy())

    connected_components_classical(binarize_img.copy())
