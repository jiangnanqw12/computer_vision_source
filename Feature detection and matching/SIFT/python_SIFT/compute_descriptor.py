import numpy as np
import cv2

# 假设已经提取了关键点，存储在变量keypoints中
# 对每个关键点生成128维的特征描述子
def generate_descriptor(image, keypoints):
    # 确定每个关键点的方向和大小
    angles, magnitudes = calculate_gradient(image)
    bins = 36
    width = 4
    descriptor_size = bins * (width ** 2)

    descriptors = []
    for kp in keypoints:
        # 获取关键点位置和方向
        x, y, angle = kp.pt[0], kp.pt[1], kp.angle
        # 创建128维的特征向量
        descriptor = np.zeros((descriptor_size,), dtype=np.float32)
        # 根据关键点方向分配每个像素点的梯度值到对应的方向直方图中
        for i in range(-width // 2, width // 2):
            for j in range(-width // 2, width // 2):
                # 确定梯度值对应的方向直方图
                bin_num = int(np.round((angle + 360) % 360 / 360 * bins))
                # 将梯度值添加到对应的直方图中
                descriptor[(i + width // 2) * bins + bin_num] = magnitudes[int(y+j), int(x+i)]
        # 对特征向量进行归一化处理
        descriptor = descriptor / np.linalg.norm(descriptor)
        # 对特征向量中的数值进行截断，避免在匹配过程中出现数值过大的情况
        descriptor = np.clip(descriptor, 0, 0.2)
        # 对特征向量再次进行归一化处理
        descriptor = descriptor / np.linalg.norm(descriptor)
        # 将特征向量添加到描述子列表中
        descriptors.append(descriptor)

    # 返回所有关键点的特征描述子
    return descriptors

# 计算图像梯度
def calculate_gradient(image):
    # 计算x、y方向的梯度值
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度大小和方向
    magnitudes = np.sqrt(gx**2 + gy**2)
    angles = np.arctan2(gy, gx) * 180 / np.pi
    return angles, magnitudes


