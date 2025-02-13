import json
import os
import shutil
from xml.etree import ElementTree as ET

import cv2
import numpy as np


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  #


def get_json_info(src_path, src_image_path, classes_name):
    # print(src_path)

    json_info = json.load(open(src_path, "r", encoding="utf-8"))

    for i, bbox in enumerate(json_info["shapes"]):
        if bbox['label'] not in classes_name:
            del json_info["shapes"][i]

    return json_info


def get_txt_info(src_path, src_image_path, classes_name):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = list(map(lambda x: x[:-1].split(" "), f.readlines()))

    txt_info = dict()
    txt_info["shapes"] = list()

    txt_info["imagePath"] = os.path.split(src_image_path)[-1]

    if os.path.isfile(src_image_path):
        img = cv2.imdecode(np.fromfile(src_image_path, dtype=np.uint8), -1)
        # cv2.imencode('.' + os.path.split(src_path)[-1], img)[1].tofile(tar_path)
        txt_info["imageHeight"], txt_info["imageWidth"], _ = img.shape
    else:
        print("图片{}不存在，需要根据图片获取图片高宽".format(src_image_path))
        # exit()
    for line in lines:
        bbox_shape = dict()

        bbox_shape["label"] = classes_name[int(line[0])]

        # print(line)
        cx = float(line[1]) * txt_info["imageWidth"]
        cy = float(line[2]) * txt_info["imageHeight"]
        w = float(line[3]) * txt_info["imageWidth"]
        h = float(line[4]) * txt_info["imageHeight"]

        xmin = cx - w / 2
        xmax = cx + w / 2
        ymin = cy - h / 2
        ymax = cy + h / 2
        # print([xmax, xmin, ymax, ymin])
        bbox_shape["points"] = [[xmax, ymax], [xmin, ymin]]

        txt_info["shapes"].append(bbox_shape)
    return txt_info


def get_xml_info(src_path, src_image_path, classes_name):
    in_file = open(src_path, "r", encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    xml_info = dict()
    xml_info["shapes"] = list()
    xml_info["imagePath"] = root.find("filename").text

    size = root.find("size")
    if int(size.find("height").text) == 0 or int(size.find("width").text) == 0:
        img = cv2.imdecode(np.fromfile(src_image_path, dtype=np.uint8), -1)
        # cv2.imencode('.jpg', img)[1].tofile(path)
        xml_info["imageHeight"], xml_info["imageWidth"], _ = img.shape
    else:
        xml_info["imageHeight"] = int(size.find("height").text)
        xml_info["imageWidth"] = int(size.find("width").text)
    if int(xml_info["imageHeight"]) == 0 or int(xml_info["imageWidth"]) == 0:
        print(src_path)
    for obj in root.iter("object"):
        # print(obj.find('name').text)
        if obj.find('name').text not in classes_name:
            # if obj.find('name').text not in classes_name.keys():
            continue

        bbox_shape = dict()

        bbox_shape["label"] = obj.find('name').text
        # bbox_shape["label"] =classes_name[obj.find('name').text]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        xmax = float(bndbox.find("xmax").text)
        ymin = float(bndbox.find("ymin").text)
        ymax = float(bndbox.find("ymax").text)

        bbox_shape["points"] = [[xmax, ymax], [xmin, ymin]]
        xml_info["shapes"].append(bbox_shape)
    return xml_info


def to_json(target_path, info, classes_name=[]):
    json_info = dict()
    json_info["version"] = "4.5.10"
    json_info["flags"] = dict()
    json_info["shapes"] = list()

    for bbox in info["shapes"]:
        json_bbox = dict()
        json_bbox["label"] = bbox["label"]
        json_bbox["points"] = bbox["points"]
        json_bbox["group_id"] = None
        json_bbox["shape_type"] = "rectangle"
        json_bbox["flags"] = dict()
        json_info["shapes"].append(json_bbox)
    json_info["imagePath"] = info["imagePath"]
    json_info["imageData"] = None
    json_info["imageHeight"] = info["imageHeight"]
    json_info["imageWidth"] = info["imageWidth"]
    # print(json_info)
    with open(target_path, "w") as f:
        f.write(json.dumps(json_info, indent=2, separators=(',', ': ')))


def to_xml(target_path, info, classes_name=[]):
    annotation = ET.Element("annotation")

    tree = ET.ElementTree(annotation)

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = info["imagePath"]

    size = ET.SubElement(annotation, 'size')

    width = ET.SubElement(size, "width")
    width.text = str(info["imageWidth"])

    height = ET.SubElement(size, "height")
    height.text = str(info["imageHeight"])
    depth = ET.SubElement(size, "depth")
    depth.text = str(3)

    for bbox_shape in info["shapes"]:  # 有多个框

        points = bbox_shape["points"]
        points = np.array(points)
        label = bbox_shape["label"]
        # print(label)
        xmin_value = min(points[:, 0])
        xmax_value = max(points[:, 0])
        ymin_value = min(points[:, 1])
        ymax_value = max(points[:, 1])
        # print(xmin_value)
        if xmax_value <= xmin_value:
            pass
        elif ymax_value <= ymin_value:
            pass
        else:
            object = ET.SubElement(annotation, "object")
            name = ET.SubElement(object, "name")
            name.text = label

            pose = ET.SubElement(object, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(object, "truncated")
            truncated.text = str(0)

            difficult = ET.SubElement(object, "difficult")
            difficult.text = str(0)

            bndbox = ET.SubElement(object, "bndbox")

            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(xmin_value))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(xmax_value))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(ymin_value))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(ymax_value))

    pretty_xml(annotation, '\t', '\n')
    tree.write(target_path, encoding="utf-8", xml_declaration=True)


def to_txt(target_path, info, classes_name=[]):
    if classes_name == []:
        print("classes_name不能为空，需要传入classes_name参数")
        return
    with open(target_path, "w") as f:
        for bbox in info["shapes"]:
            class_id = str(classes_name.index(bbox["label"]))

            points = np.array(bbox["points"])

            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])

            dw = 1.0 / info["imageWidth"]
            dh = 1.0 / info["imageHeight"]

            cx = (xmax + xmin) / 2.0
            cy = (ymax + ymin) / 2.0
            w = xmax - xmin
            h = ymax - ymin

            cx = str(cx * dw)
            cy = str(cy * dh)
            w = str(w * dw)
            h = str(h * dh)

            f.write(class_id + " " + cx + " " + cy + " " + w + " " + h + "\n")


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    classes_name = ['fire', 'candle_flame', 'round_fire', 'smoke']
    # classes_name = ['fire', 'smoke']
    # classes_name.append("del")

    src = '/home/manu/mnt/ST2000DM005-2U91/fire/data/webs/d-fire-jb/'
    tar = src
    src_label_format = "txt"
    tar_label_format = "xml"

    is_copyimg = False
    is_getnolabelimg = False

    for phase in [""]:

        tar_labels_dir = os.path.join(tar, "labels_" + tar_label_format, phase)
        makedir(tar_labels_dir)
        if is_copyimg:
            tar_images_dir = os.path.join(tar, "images", phase)
            makedir(tar_images_dir)
        if is_getnolabelimg:
            noLableDir = os.path.join(tar, "noLabelDir", phase)
            makedir(noLableDir)

        cnt_all, cnt_miss, cnt_normal = 0, 0, 0
        for image_name in os.listdir(os.path.join(src, "images", phase)):
            print(image_name)
            cnt_all += 1
            image_path = os.path.join(src, "images", phase, image_name)

            src_label_name = os.path.splitext(image_name)[0] + "." + src_label_format
            tar_label_name = os.path.splitext(image_name)[0] + "." + tar_label_format

            src_label_path = os.path.join(src, "labels_" + src_label_format, phase, src_label_name)

            if not os.path.exists(src_label_path):
                # print()
                info = []

            try:
                info = eval("get_{}_info".format(src_label_format))(src_label_path, image_path, classes_name)
                # print(info)
            except:
                info = []
            if info != [] and info["shapes"] == []:
                info = []
            # print(info)
            if info != []:
                l = [shape["label"] for shape in info["shapes"]]
                if "del" in l:
                    print(src_label_path)
                    shutil.move(image_path, "./res")
                    shutil.move(src_label_path, "./res_label")
                    continue

            if info:
                tar_label_path = os.path.join(tar, "labels_" + tar_label_format, phase, tar_label_name)
                eval("to_{}".format(tar_label_format))(tar_label_path, info, classes_name)
                cnt_normal += 1

            if is_copyimg:
                if is_getnolabelimg:
                    if info:
                        shutil.copy(image_path, tar_images_dir)
                    else:
                        shutil.copy(image_path, noLableDir)
                else:
                    shutil.copy(image_path, tar_images_dir)

    print(f'cnt_all --> {cnt_all}')
    print(f'cnt_normal --> {cnt_normal}')
