import cv2
from mmpretrain import ImageClassificationInferencer

image = '/home/manu/tmp/cls_fire_raw/pos/fire (1).mp4_combined_patch_2_1950.jpg'
# image = '/home/manu/tmp/cls_fire_raw/neg/nofire (1).mp4_combined_patch_0_175.jpg'
config = '/home/manu/mnt/ST2000DM005-2U91/workspace/mmpretrain/configs/resnet/resnet18_8xb32_fire.py'
checkpoint = '/home/manu/tmp/work_dirs/resnet18_8xb32_fire/epoch_100.pth'
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')

image = cv2.imread(image)
result = inferencer(image)[0]
print('\n')
print(result['pred_class'])
print(result['pred_scores'][1])
