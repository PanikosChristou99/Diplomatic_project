import base64
from typing import ContextManager
import flask_cors
import flask
from time import ctime
from base64 import b64decode, decode
from flask import jsonify
import fiftyone.zoo as foz
import fiftyone as fo
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import functional as func
from fiftyone import ViewField as F
from torch import device, cuda
import os
import ast
import concurrent.futures


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-val",
    max_samples=500)

classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(3)

# Load a pre-trained Faster R-CNN model
model = models.detection.retinanet_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()


app = flask.Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)

# Get the 10 most common classes in the dataset
counts = dataset.count_values("ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

# images_dir = 'images_dir'

# os.mkdir(images_dir)


@app.route('/endpoint', methods=['POST'])
def hello():
    print('Cloud got content')
    content = flask.request.get_json()
    # print('Got content')
    # print(content)
    # content2 = dict(content)
    content2 = ast.literal_eval(content)
    dataset2 = fo.Dataset("dataset-1")

    for _, dict2 in content2.items():
        sample = fo.Sample.from_dict(dict2)
        # print(sample)
        dataset2.add_sample(sample)

    for sample in dataset2:

        # base_path = os.path.basename(sample.filepath)

        # complete_name = os.path.join(images_dir, base_path)

        # image_64_decode = base64.decodebytes(sample.data)

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
        sample["retinanet_resnet"] = fo.Detections(detections=detections)
        sample.save()

    high_conf_view = dataset2.filter_labels(
        "retinanet_resnet", F("confidence") > 0.75)

    results = high_conf_view.evaluate_detections(
        "retinanet_resnet",
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
    )

    # Get the 10 most common classes in the dataset
    counts = dataset2.count_values("ground_truth.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes_top10)
    dataset2.delete()
    return jsonify(ctime())


# app.run(host='0.0.0.0', port=5001)
