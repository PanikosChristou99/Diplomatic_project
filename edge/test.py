from typing import ContextManager
from time import ctime
from base64 import b64decode, decode
from flask import jsonify
from a import content2
import fiftyone.zoo as foz
import fiftyone as fo
from PIL import Image
from torchvision import models
from torchvision.transforms import functional as func
from fiftyone import ViewField as F
from torch import device, cuda


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-val")

classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

# Load a pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Get the 10 most common classes in the dataset
counts = dataset.count_values("ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

dataset2 = fo.Dataset("dataset-1")

for _, dict in content2.items():
    dataset2.add_sample(fo.Sample(dict['filepath']))

for sample in dataset2:
    # Load image
    image = Image.open(sample.filepath)
    image = func.to_tensor(image).to(device)
    c, h, w = image.shape

    # Perform inference
    preds = model([image])[0]
    labels = preds["labels"].cpu().detach().numpy()
    scores = preds["scores"].cpu().detach().numpy()
    boxes = preds["boxes"].cpu().detach().numpy()

    # Convert detections to FiftyOne format
    detections = []
    for label, score, box in zip(labels, scores, boxes):
        # Convert to [top-left-x, top-left-y, width, height]
        # in relative coordinates in [0, 1] x [0, 1]
        x1, y1, x2, y2 = box
        rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

        detections.append(
            fo.Detection(
                label=classes[label],
                bounding_box=rel_box,
                confidence=score
            )
        )

    # Save predictions to dataset
    sample["faster_rcnn"] = fo.Detections(detections=detections)
    sample.save()

high_conf_view = dataset2.filter_labels(
    "faster_rcnn", F("confidence") > 0.75)

results = high_conf_view.evaluate_detections(
    "faster_rcnn",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)
results.print_report(classes=classes_top10)
