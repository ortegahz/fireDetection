import cv2
import numpy as np

# 创建视频捕获对象，读取视频文件
video_path = '/media/manu/ST2000DM005-2U91/fire/data/test/V3/positive/fire (1).mp4'
# video_path = '/media/manu/ST2000DM005-2U91/fire/data/test/V3/negative/nofire (33).mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否正确打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 创建背景减法器对象 (混合高斯背景建模)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果帧读取失败，结束循环
    if not ret:
        break

    # 应用背景减法
    fgmask = fgbg.apply(frame)

    # 创建一个小屏障的彩色掩码
    fg_red = np.zeros_like(frame)
    fg_red[:, :, 1] = 255

    # 将红色掩码使用前景掩码筛选出前景部分
    red_foreground = cv2.bitwise_and(fg_red, fg_red, mask=fgmask)

    # 将红色前景叠加到原始帧
    combined_frame = cv2.addWeighted(frame, 1, red_foreground, 0.5, 0)

    # 显示叠加结果
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow('Foreground Highlighted', combined_frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
