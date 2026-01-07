import os
import shutil


def process_images():
    # 1. 设置路径
    # 源文件夹路径 (默认为脚本所在目录)
    source_folder = "/media/manu/ST2000DM005-2U911/workspace/hj_smoke/data/dt20-testblur2/5米15米录制.mp4/"
    # 新建的目标文件夹名称
    target_folder_name = "processed_bmps_1080"

    target_folder_path = os.path.join("/home/manu/nfs", target_folder_name)

    # 2. 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
        print(f"已创建目标文件夹: {target_folder_path}")

    # 3. 获取源文件夹下所有文件
    files = os.listdir(source_folder)

    count = 0
    # 4. 遍历并处理文件
    for filename in files:
        # 过滤条件：
        # 1. 必须以 _ori.bmp 结尾
        # 2. 必须包含指定的前缀 "5米15米录制.mp4"
        if filename.endswith("_ori_1080.bmp") and "5米15米录制.mp4" in filename:
            try:
                # 文件名示例: 5米15米录制.mp4_0002_ori.bmp
                # 使用 '_' 进行分割
                parts = filename.split('_')

                # 分割后 parts 应该是: ['5米15米录制.mp4', '0002', 'ori.bmp']
                # 取倒数第二个元素作为索引字符串
                # index_str = parts[-2]
                index_str = parts[1]

                # 将字符串转为整数，自动去除前导0 (例如 '0002' -> 2)
                index_int = int(index_str)

                # 构建新文件名 (例如 '2.bmp')
                new_filename = f"{index_int}.bmp"

                # 构建完整的源路径和目标路径
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder_path, new_filename)

                # 复制文件 (使用 copy2 可以保留文件的时间戳等元数据)
                shutil.copy2(src_path, dst_path)

                print(f"已处理: {filename} -> {target_folder_name}/{new_filename}")
                count += 1

            except ValueError:
                print(f"跳过: {filename} (无法解析数字索引)")
            except Exception as e:
                print(f"出错: {filename} ({e})")

    print(f"\n全部完成！共提取并重命名了 {count} 张图片。")
    print(f"新文件保存在: {os.path.abspath(target_folder_path)}")


if __name__ == '__main__':
    process_images()
