import glob
import os

import cv2


def main():
    # 1. 配置路径
    input_dir = "/media/manu/ST2000DM005-2U911/workspace/hj_smoke/data/dt20-testblur2/5米15米录制.mp4/seq/"  # 当前文件夹
    output_dir = "/home/manu/tmp/output_features_sorted"  # 新建的保存文件夹

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 2. 获取所有符合 *_seqm.png 模式的文件
    # 注意：虽然提示词是 *_idx_seqm.png，但根据截图文件名是 ..._0031_seqm.png
    # 这里使用 *_seqm.png 可以匹配截图中的文件
    search_pattern = os.path.join(input_dir, "5米15米录制.mp4_*_seqm.bmp")
    files = glob.glob(search_pattern)

    if not files:
        print("未找到符合 *_seqm.png 的图片文件。")
        return

    print(f"共找到 {len(files)} 张图片，开始处理...")

    for filepath in files:
        try:
            filename = os.path.basename(filepath)

            # 3. 解析文件名中的 idx
            # 假设文件名格式为: smoke..._0031_seqm.png
            # 我们需要提取 _seqm 前面的那个数字作为 idx
            name_no_ext = os.path.splitext(filename)[0]  # 去掉 .png
            if name_no_ext.endswith('_seqm'):
                prefix = name_no_ext[:-5]  # 去掉 _seqm
                # 获取最后一个下划线后的部分作为 idx (例如 0031)
                raw_idx = prefix.split('_')[-1]
                try:
                    idx = str(int(raw_idx))
                except:
                    idx = raw_idx
            else:
                # 如果文件名格式不完全匹配，使用原文件名作为 idx 避免报错
                idx = name_no_ext

            # 4. 读取图片
            img = cv2.imread(filepath)
            if img is None:
                print(f"无法读取图片: {filename}")
                continue

            h, w, c = img.shape

            # 5. 切分图片 (上半部和下半部)
            # 高度 768，中点为 384
            mid_h = h // 2

            img_top = img[:mid_h, :]  # 上半部分 (Feature 0)
            img_bottom = img[mid_h:h, :]  # 下半部分 (Feature 1)
            # img_top = img[12:mid_h - 12, :]  # 上半部分 (Feature 0)
            # img_bottom = img[mid_h + 12:h - 12, :]  # 下半部分 (Feature 1)

            # 6. 通道交换 (BGR -> RGB)
            # OpenCV 默认读取为 BGR。
            # 你的要求是保存前转为 RGB。
            # 注意：cv2.imwrite 默认将内存数据当作 BGR 写入。
            # 如果我们将内存数据转为 RGB 后调用 imwrite，保存出来的图片颜色看起来会是反的（红蓝互换）。
            img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
            img_bottom_rgb = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2RGB)

            # 7. 构造输出文件名并保存
            # 0是上半部特征，1是下半部特征
            save_name_0 = f"img_infer_0_{idx}.bmp"
            save_name_1 = f"img_infer_1_{idx}.bmp"

            cv2.imwrite(os.path.join(output_dir, save_name_0), img_top_rgb)
            cv2.imwrite(os.path.join(output_dir, save_name_1), img_bottom_rgb)

            print(f"已处理: {filename} -> idx: {idx}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print("所有任务处理完成。")


if __name__ == "__main__":
    main()
