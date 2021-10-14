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

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        while x in self.parent:
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x < root_y:
            self.parent[y] = root_x
        elif root_x > root_y:
            self.parent[x] = root_y


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


# 8-connected classical algorithm
def connected_components_classical(img, threshold=500):
    height, width = img.shape
    labels = init_labels(img)

    # first top-down
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                labels[(y, x)] = min(find_neighbors(labels, y, x))

    # build equivalence table by UnionFind
    union_find = UnionFind()
    for y in range(height):
        for x in range(width):
            if img[y, x] == 255:
                neighbors = find_neighbors(labels, y, x)
                if len(neighbors) > 1:
                    sorted_neighbors = sorted(neighbors)
                    min_neighbor = sorted_neighbors[0]
                    for neighbor in sorted_neighbors[1:]:
                        union_find.union(min_neighbor, neighbor)

    label2pos = {}
    for pos, label in labels.items():
        root_label = union_find.find(label)

        if root_label in label2pos:
            label2pos[root_label].append(pos)
        else:
            label2pos[root_label] = [pos]

    color_img = cv2.merge([img, img, img])

    for label, positions in label2pos.items():
        if len(positions) >= threshold:
            y, x = get_centroid(positions)
            draw_cross(color_img, int(y), int(x))
            draw_bounding_box(color_img, positions)

    cv2.imwrite("connected_components.jpg", color_img)


# 8-connected iterative algorithm
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
