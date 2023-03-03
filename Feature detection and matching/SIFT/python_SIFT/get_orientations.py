import cv2
import numpy as np

def get_orientations(image, keypoints, size, sigma):
    # 计算梯度幅值和方向
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitudes = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx) * 180 / np.pi
    angles[angles < 0] += 360

    orientations = []
    for keypoint in keypoints:
        # 以关键点为中心，计算局部方向直方图
        hist = np.zeros((36,))
        x, y = keypoint.pt
        r = int(size * keypoint.size / 2)
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if x+i < 0 or x+i >= image.shape[1] or y+j < 0 or y+j >= image.shape[0]:
                    continue
                bin_num = int(np.round(angles[int(y+j), int(x+i)] / 10) % 36)
                hist[bin_num] += magnitudes[int(y+j), int(x+i)]

        # 寻找局部方向直方图中的峰值作为主方向
        max_bin = np.argmax(hist)
        max_value = hist[max_bin]
        for bin_num in range(len(hist)):
            # 大于80%的峰值作为次方向
            if bin_num != max_bin and hist[bin_num] >= 0.8 * max_value:
                orientations.append((keypoint.pt, (bin_num + 0.5) * 10))

        # 如果没有次方向，则以主方向为唯一方向
        if len(orientations) == 0:
            orientations.append((keypoint.pt, (max_bin + 0.5) * 10))

    return orientations
