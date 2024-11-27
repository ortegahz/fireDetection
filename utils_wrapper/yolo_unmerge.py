import os
import shutil

from tqdm import tqdm  # 引入tqdm

from utils import make_dirs

imgs_folder = '/home/manu/mnt/8gpu_3090/test/fire/images/pseudof'
txts_folder = '/home/manu/mnt/8gpu_3090/test/fire/labels/pseudof'
output_folder = '/home/manu/tmp/yolo_unmerge'

make_dirs(output_folder, reset=True)

txt_files = [f for f in os.listdir(txts_folder) if f.endswith('.txt')]

for txt_file in tqdm(txt_files, desc="Processing files", unit="file"):
    parts = txt_file.split('_frame_')
    if len(parts) < 2:
        continue
    videoname = parts[0]

    video_output_folder = os.path.join(output_folder, videoname)
    make_dirs(video_output_folder, reset=False)

    for img_extension in ['.jpg', '.png', '.jpeg']:
        img_file = parts[0] + '_frame_' + parts[1].replace('.txt', img_extension)
        img_path = os.path.join(imgs_folder, img_file)

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(video_output_folder, img_file))
            break

    txt_path = os.path.join(txts_folder, txt_file)
    if os.path.exists(txt_path):
        shutil.copy(txt_path, os.path.join(video_output_folder, txt_file))
