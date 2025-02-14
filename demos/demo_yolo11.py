from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
# model = YOLO("/home/manu/tmp/runs_yolo11/train13/weights/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

print(results)

dets = results[0]  # single img
for cls, box, conf in zip(dets.boxes.cls.cpu().numpy(),
                          dets.boxes.xywhn.cpu().numpy(),
                          dets.boxes.conf.cpu().numpy()):
    center_x, center_y, box_w, box_h = box
    print((cls, box, conf))
    print((center_x, center_y, box_w, box_h))
