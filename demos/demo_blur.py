import cv2

# 创建视频捕捉对象，从文件读取视频
video_path = '/home/manu/mnt/ST8000DM004-2U91/smoke/data/test/烟雾/正例（200）/smog (56).mp4'
# video_path = '/home/manu/mnt/ST8000DM004-2U91/smoke/data/test/烟雾/反例（200）/nosmog (10).mp4'
cap = cv2.VideoCapture(video_path)

# 创建背景分割器对象
backSub = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)

# 初始化前一个帧
prev_frame = None

while True:
    # 读取视频的每一帧
    ret, frame = cap.read()

    # 检查视频是否结束
    if not ret:
        break

    # 使用背景分割器获取前景掩码
    fgMask = backSub.apply(frame)

    # 计算帧差
    if prev_frame is not None:
        frame_diff = cv2.absdiff(frame, prev_frame)
    else:
        frame_diff = frame

    prev_frame = frame.copy()

    # 创建和调整窗口大小
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Frame Difference', cv2.WINDOW_NORMAL)

    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('FG Mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Frame Difference', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 显示当前帧、前景掩码和帧差
    cv2.imshow('Frame', frame)
    # cv2.imshow('FG Mask', fgMask)
    cv2.imshow('Frame Difference', frame_diff * 16)

    # 按下 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
