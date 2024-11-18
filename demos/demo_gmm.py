import cv2

# 创建视频捕获对象，读取视频文件
video_path = '/media/manu/ST2000DM005-2U91/fire/data/20240806/BOSH-FM数据采集/zheng-shinei/Z-D-170m-2.mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否正确打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 创建背景减法器对象 (混合高斯背景建模)
fgbg = cv2.createBackgroundSubtractorMOG2(history=512, varThreshold=16, detectShadows=False)

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果帧读取失败，结束循环
    if not ret:
        break

    # 转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 提取H通道
    h_channel = hsv[:, :, 0]

    # 提取原始图像的R通道
    r_channel = frame[:, :, 2]

    # 计算前景掩码
    fgmask = fgbg.apply(frame)

    # 将前景掩码扩展为3通道
    fg_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    # 合成RGB图像（这里我们用R通道, 前景掩码, 和H通道分别作为R, G, B通道）
    combined_rgb = cv2.merge([r_channel, fg_rgb[:, :, 1], h_channel])

    # 显示结果
    cv2.imshow('Combined RGB', combined_rgb)

    # 按 'q' 键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
