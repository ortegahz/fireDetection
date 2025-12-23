from io import BytesIO

import requests
import supervision as sv
from PIL import Image
from inference import get_model

# url = "https://media.roboflow.com/dog.jpeg"
# image = Image.open(BytesIO(requests.get(url).content))

image = Image.open("/home/manu/图片/vlcsnap-2025-12-22-16h39m24s029.png")

model = get_model("rfdetr-medium")

predictions = model.infer(image, confidence=0.2)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)