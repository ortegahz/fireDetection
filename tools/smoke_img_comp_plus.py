import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def compare_images(img1_path, img2_path, show_plot=True):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if img1.shape != img2.shape:
        raise ValueError("Images must be the same size.")

    # 基础差异计算
    diff = np.abs(img1 - img2)  # pixel absolute difference (per channel)
    diff_gray = diff.mean(axis=2)  # convert to grayscale difference (0-255)

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

    print(f"最大像素差异: {max_diff:.4f}")
    print(f"最小像素差异: {min_diff:.4f}")
    print(f"平均误差 (Mean Diff): {mean_diff:.4f}")
    print(f"方差 (Variance): {variance:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"峰值信噪比 (PSNR): {psnr:.4f} dB")

    if show_plot:
        plt.figure(figsize=(12, 6))  # 增加高度以容纳文本

        # Image 1
        plt.subplot(1, 3, 1)
        plt.title("Image 1")
        plt.imshow(img1.astype(np.uint8))
        plt.axis("off")

        # Image 2
        plt.subplot(1, 3, 2)
        plt.title("Image 2")
        plt.imshow(img2.astype(np.uint8))
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
    dir_a = "/home/manu/tmp/output_features_sorted/"
    dir_b = "/home/manu/nfs/smoke_infer_input/"

    if os.path.exists(dir_a) and os.path.exists(dir_b):
        # 获取文件夹a中所有符合 img_infer_1_*.bmp 的文件
        files = [f for f in os.listdir(dir_a) if f.startswith("img_infer_0_") and f.endswith(".bmp")]

        # 尝试按索引数字排序 (假设格式固定为 img_infer_1_{idx}.bmp)
        try:
            files.sort(key=lambda x: int(x.replace("img_infer_0_", "").replace(".bmp", "")))
        except ValueError:
            files.sort()  # 如果文件名格式不完全符合预期，回退到默认字符串排序

        all_metrics = []

        for filename in files:
            path_a = os.path.join(dir_a, filename)
            path_b = os.path.join(dir_b, filename)

            if os.path.exists(path_b):
                print(f"\n>>> Comparing: {filename}")
                try:
                    # 批量处理时通常关闭绘图 (show_plot=False)，以免弹出大量窗口
                    # 接收返回的指标元组
                    metrics = compare_images(path_a, path_b, show_plot=False)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"\n>>> Missing in target folder: {filename}")

        # 循环结束后打印整体统计信息
        if all_metrics:
            # 转换为numpy数组方便计算列的统计值
            # indices: 0:max, 1:min, 2:mean, 3:var, 4:mse, 5:psnr
            data = np.array(all_metrics)

            print("\n" + "=" * 40)
            print("         Overall Statistics")
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
