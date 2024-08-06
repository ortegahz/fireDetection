# infer

export CUDA_VISIBLE_DEVICES=0

python detect.py --source ./data/images/horses.jpg --img 640 --device 0 --weights /home/Huangzhe/Test/yolov9-c-converted.pt --name yolov9_c_c_640_detect

python detect_dual.py --source /home/manu/tmp/BOSH-FM数据采集/xiang/X-170m-002.mp4 --img 1280 --device 0 --weights /run/user/1000/gvfs/smb-share:server=172.20.254.200,share=sharedfolder/Test/yolov9-s-fire-12809/weights/best.pt --name yolov9_s_c_1280_detect --view-img --conf-thres 0.25

