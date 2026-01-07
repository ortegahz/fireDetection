import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def compare_images(img1_path, img2_path, channel='ALL', show_plot=True):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if img1.shape != img2.shape:
        raise ValueError("Images must be the same size.")

    # 通道选择逻辑 (PIL: R=0, G=1, B=2)
    if channel != 'ALL':
        channel_map = {'R': 0, 'G': 1, 'B': 2}
        if channel not in channel_map:
            raise ValueError("Channel must be 'R', 'G', 'B' or 'ALL'")
        idx = channel_map[channel]
        img1 = img1[:, :, idx]
        img2 = img2[:, :, idx]

    # 基础差异计算
    diff = np.abs(img1 - img2)  # pixel absolute difference (per channel)
    # 如果是单通道，diff本身就是二维的；如果是ALL，需要求均值转灰度
    diff_gray = diff if channel != 'ALL' else diff.mean(axis=2)

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

    print(f"--- Channel: {channel} ---")
    print(f"最大像素差异: {max_diff:.4f}")
    print(f"最小像素差异: {min_diff:.4f}")
    print(f"平均误差 (Mean Diff): {mean_diff:.4f}")
    print(f"方差 (Variance): {variance:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"峰值信噪比 (PSNR): {psnr:.4f} dB")

    if show_plot:
        plt.figure(figsize=(12, 6))  # 增加高度以容纳文本

        cmap_opt = 'gray' if channel != 'ALL' else None

        # Image 1
        plt.subplot(1, 3, 1)
        plt.title(f"Image 1 ({channel})")
        plt.imshow(img1.astype(np.uint8), cmap=cmap_opt)
        plt.axis("off")

        # Image 2
        plt.subplot(1, 3, 2)
        plt.title(f"Image 2 ({channel})")
        plt.imshow(img2.astype(np.uint8), cmap=cmap_opt)
        plt.axis("off")

        # Difference Heatmap
        plt.subplot(1, 3, 3)
        plt.title("Difference Heatmap")
        plt.imshow(diff_gray, cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        # 将指标绘制在图表下方
        metrics_text = (
            f"Max Diff: {max_diff:.2f} | Min Diff: {min_diff:.2f}\n"
            f"Mean Diff: {mean_diff:.2f} | Variance: {variance:.2f}\n"
            f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB"
        )

        # 使用 figtext 在底部居中添加文本框
        plt.figtext(0.5, 0.05, metrics_text, ha="center", fontsize=12,
                    bbox={"facecolor": "#f0f0f0", "alpha": 0.8, "pad": 10, "edgecolor": "gray"})

        # 调整布局，给底部文本留出空间
        plt.subplots_adjust(bottom=0.25)

        plt.show()

    return max_diff, min_diff, mean_diff, variance, mse, psnr


if __name__ == "__main__":
    # 示例调用
    # 请确保路径存在，或者替换为您本地的图片路径
    try:
        compare_images(
            # "/home/manu/tmp/output_features_sorted/img_infer_0_5.bmp",
            # "/home/manu/nfs/smoke_infer_input_v0/img_infer_0_5.bmp",
            # "/home/manu/tmp/bg_image_base_6.bmp",
            # "/home/manu/nfs/backgroundDstGray_6.bmp",
            # "/home/manu/tmp/base.bmp",
            # "/home/manu/nfs/backgroundDstGray_7.bmp",
            # "/home/manu/tmp/channelA.bmp",
            # "/home/manu/nfs/channelA.bmp",
            "/home/manu/nfs/smoke_input_v5/img_1.bmp",
            "/home/manu/nfs/smoke_input/img_1.bmp",
            channel="B",  # 这里指定比对 B 通道
            show_plot=True)
    except FileNotFoundError:
        print("示例图片路径不存在，请修改路径后重试。")
