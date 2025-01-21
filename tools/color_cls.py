import os

import cv2

# 指定包含图像的文件夹路径
folder_path = '/home/manu/tmp/rgb2jpg/'  # 替换为你的文件夹路径

# 列出文件夹中所有的图像文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 遍历每个图像文件并进行处理
for image_file in image_files:
    # 构建每个文件的完整路径
    image_path = os.path.join(folder_path, image_file)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像文件: {image_file}")
        continue

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建掩膜，只包含灰度值低于阈值的区域
    threshold_value = 50
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 复制原始图像用于绘制
    output_image = image.copy()

    # 将掩膜上的点画成黄点
    yellow_color = (0, 255, 0)  # BGR格式中的黄色
    output_image[mask == 255] = yellow_color

    # 显示结果
    cv2.imshow('Original Image with Yellow Dots', output_image)

    # 等待按键，按任意键后显示下一张图像
    print(f"Displaying {image_file}. Press any key to continue...")
    cv2.waitKey(500)

cv2.destroyAllWindows()
