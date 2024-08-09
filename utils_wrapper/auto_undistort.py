import glob
import random

import cv2
import numpy as np

# 准备特征点检测和匹配器
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 读取图像
images = glob.glob('/home/manu/tmp/BOSH-FM数据采集-samples-merge/*.jpg')

# 随机选择30到100张图片
selected_images = random.sample(images, 4096)

# 检测和匹配特征点
obj_points = []
img_points = []

for i in range(len(selected_images) - 1):
    img1 = cv2.imread(selected_images[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(selected_images[i + 1], cv2.IMREAD_GRAYSCALE)

    # 检测特征点和计算描述子
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 匹配特征点
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配的特征点
    img1_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    img2_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(img1_pts, img2_pts, cv2.FM_RANSAC)

    # 只保留内点
    img1_pts = img1_pts[mask.ravel() == 1]
    img2_pts = img2_pts[mask.ravel() == 1]

    # 添加到集合中
    if len(img1_pts) > 0 and len(img2_pts) > 0:
        obj_points.append(np.hstack([img1_pts, np.zeros((img1_pts.shape[0], 1))]).reshape(-1, 1, 3))
        img_points.append(img1_pts.reshape(-1, 1, 2))

# 将特征点整理为相机校正所需的格式
obj_points = [np.array(pts, dtype=np.float32) for pts in obj_points]
img_points = [np.array(pts, dtype=np.float32) for pts in img_points]

# 相机校正
ret, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.calibrateCamera(obj_points, img_points, img1.shape[::-1], None, None)

print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)

# 纠正畸变
img = cv2.imread('/home/manu/tmp/BOSH-FM数据采集-samples-merge/jiu-shinei_J-D-10m-001_frame_000000.jpg')
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# 纠正畸变的图像
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
