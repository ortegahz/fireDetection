import io
import os
import cv2
from datetime import datetime
import requests
import numpy as np
import time
from loguru import logger
import random
logger.add("sfhPush1.log", format="{time} {level} {message}", level="DEBUG")#, rotation="200MB", compression="zip")
host = "http://172.20.20.188:8080"
#host = "http://192.168.1.90:8080"


#list_one= [30,33,35,38,40]
list_one = [20]
def push_video(video_path):
    url = host + "/fireDetect/JBFuploadTest"
    cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    black_img = np.zeros((1080, 1920, 3),dtype=np.uint8)
    cv2.imshow("test",black_img)
    cv2.waitKey(1)

    total_count, count, total_sum = 0, 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("{}视频不存在".format(video_path))
        exit()
    jpg_count = 0


    img = cv2.imread("/home/manu/nfs/output_frames/1.png")
    q_img = img[:,:,::-1]

    for i in range(30):
        dst_byte = q_img.tobytes()
        buffered_reader = io.BytesIO(dst_byte)
        file_like_object = io.BufferedReader(buffered_reader)
        black_dst_byte = black_img.tobytes()
        black_buffered_reader = io.BytesIO(black_dst_byte)
        black_like_object = io.BufferedReader(black_buffered_reader)

        files = {
                "Ir": black_like_object
        }
        files["Visi"] = file_like_object

        try:
            start = time.time()
            response = requests.post(url, files=files, verify=False)
            cur_time = round(time.time() - start,3)
        except Exception as e:
            logger.debug("推理次数:{},耗时{},接口访问失败:{}".format(total_count,cur_time,e))



    while True:
        success, img = cap.read()
        total_sum += 1
        if not success:
            logger.info("{}视频测试完毕".format(video_path))
            break
        time_tf = random.choice(list_one)
        if total_sum % time_tf == 0:
            dst_img = img
            dst_img = cv2.resize(img, (1920, 1080))
            jpg_count += 1

            # if jpg_count == 1:
            #     cv2.imwrite("/home/bossliu/workspace/testdata/one1.jpg", dst_img)

            rgb_img = dst_img[:,:,::-1]
            dst_byte = rgb_img.tobytes()
            buffered_reader = io.BytesIO(dst_byte)
            file_like_object = io.BufferedReader(buffered_reader)
            # 读取数据

            black_dst_byte = black_img.tobytes()
            black_buffered_reader = io.BytesIO(black_dst_byte)
            black_like_object = io.BufferedReader(black_buffered_reader)

            files = {
                "Ir": black_like_object
            }


            files["Visi"] = file_like_object
            #files["Ir"] = file_like_object
            try:
                start = time.time()
                response = requests.post(url, files=files, verify=False)
                cur_time = round(time.time() - start,3)
            except Exception as e:
                logger.debug("推理次数:{},耗时{},接口访问失败:{}".format(total_count,cur_time,e))
            total_count += 1
            response_json = response.json()
            if 0 < response_json["code"] < 9:
                count += 1
                logger.debug("推理次数:{},耗时{},有报警产生".format(total_count, cur_time))
                logger.debug(response_json)
                bboxes = response_json.get("smog")
                if bboxes:
                    bbox = bboxes["visi"][0]
                    x1 = bbox["x"]
                    y1 = bbox["y"]
                    x2 = bbox["x"] + bbox["w"]
                    y2 = bbox["y"] + bbox["h"]
                    cv2.rectangle(dst_img, (x1, y1), (x2, y2), (255,255,0), 2)
            else:
                logger.info("推理次数:{},耗时{},无报警产生".format(total_count, cur_time))
            cv2.imshow("test",dst_img)
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    logger.debug(f"{video_path} 报警数：{count}, 总数：{total_count}, 报警比例：{count / total_count}")







if __name__ == '__main__':
    url = host + "/fireDetect/setSampleDetectSwitch"

    payload = {"sampleDetectSwitch": 1}
    headers = {"content-type": "application/json"}

    response = requests.request("POST", url, json=payload, headers=headers, timeout=5, verify=False)
    time.sleep(5)
    logger.info(response.text)
    video_path = "/media/manu/ST8000DM004-2U91/tmp/DT的VLC烟雾/5米15米录制.mp4" #"/home/bossliu/workspace/camera/20251205/visiDir_1764925433/visi_1764925436.ts" #"/home/bossliu/100m.mp4"
    push_video(video_path)
    payload = {"sampleDetectSwitch": 0}
    response = requests.request("POST", url, json=payload, headers=headers, timeout=5, verify=False)
    time.sleep(5)
    logger.info(response.text)
