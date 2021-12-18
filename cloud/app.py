import flask_cors
import flask
from time import ctime
from flask import jsonify
import fiftyone.zoo as foz
import fiftyone as fo
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import functional as func
from fiftyone import ViewField as F
from torch import device, cuda
from bson.json_util import loads

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-val",
    max_samples=500)

# Get all classes the dataset contains
classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(3)

# Load a resnet50 model
model = models.detection.retinanet_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()


app = flask.Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)


@app.route('/endpoint', methods=['POST'])
def hello():
    try:
        print('Cloud got content')
        content = flask.request.get_json()

        # Create the dict of 2 dicts from the json sent
        content2 = loads(content)

        # Create the fiftyone dataset dict
        source = loads(content2['contents'])

        # first_value = list(source.values())[0]
        # print('First Value: ', first_value)

        # Create the previous results dictionary
        results_prev = content2['results']

        # second_value = list(results_prev.values())[0]
        # print('Second Value: ', second_value)

        dataset2 = fo.Dataset()

        # Fill the dataset with the data recieved
        for _, dict2 in source.items():
            sample = fo.Sample.from_dict(dict2)
            # print(sample)
            dataset2.add_sample(sample)

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
            sample["retinanet_resnet"] = fo.Detections(detections=detections)
            sample["faster_rcnn"] = fo.Detections.from_dict(
                results_prev[sample.filepath])
            sample.save()

        # Get the predictions we are only really confident about
        high_conf_view = dataset2.filter_labels(
            "retinanet_resnet", F("confidence") > 0.75)

        # Evaluate if we were correct
        results1 = high_conf_view.evaluate_detections(
            "retinanet_resnet",
            gt_field="ground_truth",
            eval_key="eval1",
            compute_mAP=True,
        )

        # Get the 10 most common classes in the dataset
        counts = dataset2.count_values("ground_truth.detections.label")
        classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

        # Print a classification report for the top-10 classes
        results1.print_report(classes=classes_top10)

        high_conf_view2 = dataset2.filter_labels(
            "faster_rcnn", F("confidence") > 0.75)

        results2 = high_conf_view2.evaluate_detections(
            "faster_rcnn",
            gt_field="ground_truth",
            eval_key="eval2",
            compute_mAP=True,
        )

        # Print a classification report for the top-10 classes
        results2.print_report(classes=classes_top10)

        # results.
        # to_send = {}

        # asyncio.get_event_loop().run_in_executor(
        #     None, send_to_db, content2)  # fire and forget

        return jsonify(ctime())
    except Exception as e:
        print(e)
        return jsonify(ctime())

# app.run(host='0.0.0.0', port=5001)
