import cv2
import numpy as np


def main():
    # 创建背景减除器
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # 打开视频文件或摄像头
    video_path = '/home/manu/mnt/ST8000DM004-2U91/jade_raw_data/03数据标注/01 数据采集/bosch数据采集/BOSH-FM数据采集/BOSH-FM数据采集/zheng-shinei/Z-D-170m-2.mp4'  # 如果要使用摄像头，将其设置为0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用背景减除算法
        fg_mask = background_subtractor.apply(frame)

        # 对前景掩码进行一些后处理
        # 去噪声操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 查找前景中的轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 8:  # 忽略小面积的物体
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 将前景掩码转换为红色通道
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :, 2] = fg_mask  # 将掩码应用到红色通道

        # 将前景掩码叠加到原始图像上
        combined = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

        # 显示结果
        cv2.imshow('Frame with Mask', combined)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
