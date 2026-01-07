import os

import cv2


def extract_frames(video_path, output_folder):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return

    # 保存文件的序号计数器，从1开始
    save_index = 1

    # --- 第一阶段：处理第一帧 ---
    ret, first_frame = cap.read()
    if not ret:
        print("Error: 无法读取第一帧")
        return

    _img_ext = '.bmp'

    print("正在处理第一帧 (重复保存30次)...")
    # 将第一帧连续保存30次 (1.jpg - 30.jpg)
    for _ in range(30):
        filename = os.path.join(output_folder, f"{save_index}{_img_ext}")
        cv2.imwrite(filename, first_frame)
        save_index += 1

    # --- 第二阶段：处理后续帧 ---
    print("正在处理后续帧 (每20帧保存一次)...")

    # 当前视频流中的帧计数
    # 因为已经读取了第0帧(first_frame)，所以现在循环从第1帧开始
    video_frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 判断是否满足每20帧保存一次的条件
        # 这里假设是相对于视频开头的绝对帧数 (如第20帧, 第40帧...)
        if video_frame_count % 20 == 0:
            filename = os.path.join(output_folder, f"{save_index}{_img_ext}")
            cv2.imwrite(filename, frame)
            # print(f"已保存: {filename} (原视频帧号: {video_frame_count})")
            save_index += 1

        video_frame_count += 1

    cap.release()
    print(f"处理完成。共保存了 {save_index - 1} 张图片。")


if __name__ == "__main__":
    # 在此处修改视频路径和输出文件夹
    VIDEO_FILE = "/media/manu/ST8000DM004-2U91/tmp/DT的VLC烟雾/5米15米录制.mp4"
    OUTPUT_DIR = "/home/manu/nfs/output_frames_bmp"

    extract_frames(VIDEO_FILE, OUTPUT_DIR)
