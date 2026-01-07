import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def compare_images(img1_path, img2_path, show_plot=True, channel=None):
    """
    比对两张图片。
    :param channel: 指定比对通道，可选 'R', 'G', 'B' 或 None (默认比对所有通道)
    """
    img1_full = load_image(img1_path)
    img2_full = load_image(img2_path)

    if img1_full.shape != img2_full.shape:
        raise ValueError("Images must be the same size.")

    # 处理通道选择
    channel_name = "RGB"
    if channel is not None:
        channel_map = {'R': 0, 'G': 1, 'B': 2}
        if channel in channel_map:
            idx = channel_map[channel]
            # 提取单通道，变为二维数组 (H, W)
            img1 = img1_full[:, :, idx]
            img2 = img2_full[:, :, idx]
            channel_name = f"{channel} Channel"
        else:
            raise ValueError("Channel must be 'R', 'G', 'B' or None")
    else:
        # 使用完整图像
        img1 = img1_full
        img2 = img2_full

    # 基础差异计算
    diff = np.abs(img1 - img2)

    # 如果是单通道(2D)，差异本身就是灰度差异；如果是多通道(3D)，需要求均值转为灰度
    if diff.ndim == 2:
        diff_gray = diff
    else:
        diff_gray = diff.mean(axis=2)

    # 统计指标
    max_diff = diff_gray.max()
    min_diff = diff_gray.min()
    mean_diff = diff_gray.mean()
    variance = diff_gray.var()

    # 新增指标：MSE (均方误差)
    mse = np.mean((img1 - img2) ** 2)

    # 新增指标：PSNR (峰值信噪比)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    print(f"[{channel_name}] 最大差异: {max_diff:.4f}")
    print(f"[{channel_name}] 最小差异: {min_diff:.4f}")
    print(f"[{channel_name}] 平均误差: {mean_diff:.4f}")
    print(f"[{channel_name}] MSE: {mse:.4f}")
    print(f"[{channel_name}] PSNR: {psnr:.4f} dB")

    if show_plot:
        plt.figure(figsize=(12, 6))

        # 确定显示用的 colormap，如果是单通道则用灰度显示
        cmap_img = 'gray' if img1.ndim == 2 else None

        # Image 1
        plt.subplot(1, 3, 1)
        plt.title(f"Image 1 ({channel_name})")
        plt.imshow(img1.astype(np.uint8), cmap=cmap_img)
        plt.axis("off")

        # Image 2
        plt.subplot(1, 3, 2)
        plt.title(f"Image 2 ({channel_name})")
        plt.imshow(img2.astype(np.uint8), cmap=cmap_img)
        plt.axis("off")

        # Difference Heatmap
        plt.subplot(1, 3, 3)
        plt.title("Difference Heatmap")
        plt.imshow(diff_gray, cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        # 将指标绘制在图表下方
        metrics_text = (
            f"Channel: {channel_name}\n"
            f"Max Diff: {max_diff:.2f} | Min Diff: {min_diff:.2f}\n"
            f"Mean Diff: {mean_diff:.2f} | Variance: {variance:.2f}\n"
            f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB"
        )

        plt.figtext(0.5, 0.05, metrics_text, ha="center", fontsize=12,
                    bbox={"facecolor": "#f0f0f0", "alpha": 0.8, "pad": 10, "edgecolor": "gray"})

        plt.subplots_adjust(bottom=0.25)
        plt.show()

    return max_diff, min_diff, mean_diff, variance, mse, psnr


if __name__ == "__main__":
    dir_a = "/home/manu/tmp/output_features_sorted/"
    dir_b = "/home/manu/nfs/smoke_infer_input/"

    # ================= 配置区域 =================
    # 设置要比对的通道：'R', 'G', 'B' 或 None (None表示比对全图)
    target_block = 0
    target_channel = 'B'
    # target_block = 1
    # target_channel = None
    # ===========================================

    if os.path.exists(dir_a) and os.path.exists(dir_b):
        files = [f for f in os.listdir(dir_a) if f.startswith(f"img_infer_{target_block}_") and f.endswith(".bmp")]

        try:
            files.sort(key=lambda x: int(x.replace(f"img_infer_{target_block}_", "").replace(".bmp", "")))
        except ValueError:
            files.sort()

        all_metrics = []

        print(f"Starting comparison on channel: {target_channel if target_channel else 'ALL (RGB)'}")

        for filename in files:
            path_a = os.path.join(dir_a, filename)
            path_b = os.path.join(dir_b, filename)

            if os.path.exists(path_b):
                print(f"\n>>> Comparing: {filename}")
                try:
                    metrics = compare_images(path_a, path_b, show_plot=False, channel=target_channel)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"\n>>> Missing in target folder: {filename}")

        if all_metrics:
            data = np.array(all_metrics)
            print("\n" + "=" * 40)
            print(f"         Overall Statistics ({target_channel if target_channel else 'RGB'})")
            print("=" * 40)
            print(f"Total Images Compared : {len(all_metrics)}")
            print(f"Overall Max Diff      : {np.max(data[:, 0]):.4f}")
            print(f"Overall Min Diff      : {np.min(data[:, 1]):.4f}")
            print(f"Average Mean Diff     : {np.mean(data[:, 2]):.4f}")
            print(f"Average MSE           : {np.mean(data[:, 4]):.4f}")
            print(f"Average PSNR          : {np.mean(data[:, 5]):.4f} dB")
            print("=" * 40)
        else:
            print("\nNo images were compared successfully.")

    else:
        print(f"Directory not found:\n{dir_a}\nor\n{dir_b}")
