from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url

download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png", "demo_data/terrain2.png"
)
download_from_url("https://ultralytics.com/images/boats.jpg", "demo_data/obb_test_image.png")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11s.pt",  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
    confidence_threshold=0.1,
    device="cuda:0",  # or 'cuda:0' if GPU is available
)

# result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

# result = get_prediction(read_image("/home/manu/图片/vlcsnap-2025-12-22-16h39m24s029.png"), detection_model)

# result.export_visuals(export_dir="demo_data/", hide_conf=True)
#
# Image("demo_data/prediction_visual.png")

result = get_sliced_prediction(
    "/home/manu/图片/vlcsnap-2025-12-22-16h39m24s029.png",
    detection_model,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)


# Filter results to keep only "person" class
result.object_prediction_list = [
    pred for pred in result.object_prediction_list if pred.category.name == "person"
]

result.export_visuals(export_dir="demo_data/", hide_conf=True)

Image("demo_data/prediction_visual.png")
