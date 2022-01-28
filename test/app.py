from base64 import b64decode, b64encode
import json
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
from torch import device, cuda
import ast
from helper_edge import send_to_cloud
import asyncio
from os import environ, getcwd, path, remove
from fiftyone import ViewField as F
import io
import warnings
warnings.filterwarnings("ignore")

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

torch.set_num_threads(1)


# Create model pointer
model = None
model_name = ""

# if we have a custom ML then use that
if 'ML' in environ:
    model_name = environ['ML']
    method = getattr(models.detection, model_name)
    model = method(pretrained=True)
    print('Using ', model_name)
else:
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model_name = environ['fasterrcnn_resnet50_fpn']
    print('Using ', model_name)


model.to(device)
model.eval()

# Dictionary with the picture encoded in base64, as well as all attributes the image has like what it contains, dimensions etc. (What fiftyone natively has)
dicts = {}
# This dict is the same as above but without the picture encoded, I print this some tims to debug stuff
dict_no_data = {}

predictions_view = dataset.take(5, seed=51)

for sample in predictions_view:
    with open(sample.filepath, "rb") as image:
        # Add to the smaller the dct the data for thsi sample
        dict_no_data[sample['id']] = sample.to_dict()
        # Write the base64 encoded image to the Sample
        sample['data'] = b64encode(image.read()).decode('utf-8')
        # Add the sample to the dict to be sent as workload
        dicts[sample['id']] = sample.to_dict()

    # Dump the dictionary as a string
dictToSend = json.dumps(dicts)
content2 = ast.literal_eval(dictToSend)
dataset2 = fo.Dataset()

# Create the fiftyone dataset dict
for _, dict2 in content2.items():
    sample = fo.Sample.from_dict(dict2)
    # print(sample)
    dataset2.add_sample(sample)

results_dict = {}

for sample in dataset2:

    # Load image

    # image_data = b64decode(sample.data)
    # dec_data = open(sample.data, mode="r", encoding="utf-8")
    image_data = b64decode(sample.data)
    image = Image.open(io.BytesIO(image_data))

    if 'Preprocessing' in environ:

        curr_path = path.abspath(getcwd())
        pic_name = path.basename(sample.filepath)
        new_path = path.join(curr_path, pic_name)

        image.save(new_path, quality=100, optimize=True)
        print("Original image size = ", path.getsize(new_path) / (1024*1024))
        remove(new_path)

        preferences_str = environ['Preprocessing']
        # print(preferences_str)

        # catch if someone forgot coma at the end
        if preferences_str[-1] == ',':
            preferences_str = preferences_str[:-1]

        pref_arr = preferences_str.split(',')
        pref_names = pref_arr[::2]
        # print(pref_names)

        pref_values = pref_arr[1::2]
        # print(pref_values)

        pref_dict = {}
        for i, name in enumerate(pref_names):
            pref_dict[name] = pref_values[i]

        # print(json.dumps(pref_dict, sort_keys=True, indent=4))

        resize = 100

        if 'resize' in pref_dict:
            resize = pref_dict['resize']
            if resize[-1] == '%':
                resize = resize[:-1]
            resize = float(resize)/100

        s = image.size
        new0 = float(s[0])*resize
        new1 = float(s[1])*resize
        image = image.resize((int(new0), int(new1)))

        quality = 100

        if 'quality' in pref_dict:
            quality = pref_dict['quality']
            if quality[-1] == '%':
                quality = quality[:-1]
            quality = int(quality)

        curr_path = path.abspath(getcwd())
        pic_name = path.basename(sample.filepath)

        image.save(new_path, quality=quality, optimize=True)
        print("New image size = ", path.getsize(new_path) / (1024*1024))

        image = Image.open(new_path)
        remove(new_path)

        if 'BW' in pref_dict:
            if pref_dict['BW'] != '0':
                image.draft("L", image.size)

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
        sample[model_name] = fo.Detections(detections=detections)
        sample.save()
        results_dict[sample.filepath] = dict(fo.Detections(
            detections=detections).to_dict())

    # Uncomment the below to print the report for this only


high_conf_view = dataset2.filter_labels(
    model_name, F("confidence") > 0.75)

results = high_conf_view.evaluate_detections(
    model_name,
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)

# Get the 10 most common classes in the dataset
counts = dataset2.count_values("ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 classes
results.print_report(classes=classes_top10)


# app = flask.Flask(__name__)
# # This allows for running the app and taking in requests from the same computer
# flask_cors.CORS(app)


# @app.route('/endpoint', methods=['POST'])
# async def hello():
#     try:
#         print('Edge got content')
#         content = flask.request.get_json()

#         # create the dict from the json sent
#         content2 = ast.literal_eval(content)
#         dataset2 = fo.Dataset()

#         # Create the fiftyone dataset dict
#         for _, dict2 in content2.items():
#             sample = fo.Sample.from_dict(dict2)
#             # print(sample)
#             dataset2.add_sample(sample)

#         results_dict = {}

#         for sample in dataset2:

#             # Load image
#             image = Image.open(sample.filepath)
#             image = func.to_tensor(image).to(device)
#             c, h, w = image.shape

#             # Perform inference
#             preds = model([image])[0]
#             labels = preds["labels"].cpu().detach().numpy()
#             scores = preds["scores"].cpu().detach().numpy()
#             boxes = preds["boxes"].cpu().detach().numpy()

#             # Convert detections to FiftyOne format
#             detections = []
#             for label, score, box in zip(labels, scores, boxes):
#                 # Convert to [top-left-x, top-left-y, width, height]
#                 # in relative coordinates in [0, 1] x [0, 1]
#                 x1, y1, x2, y2 = box
#                 rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

#                 detections.append(
#                     fo.Detection(
#                         label=classes[label],
#                         bounding_box=rel_box,
#                         confidence=score
#                     )
#                 )

#             # Save predictions to dataset
#             sample["faster_rcnn"] = fo.Detections(detections=detections)
#             sample.save()
#             results_dict[sample.filepath] = dict(fo.Detections(
#                 detections=detections).to_dict())

#         # Uncomment the below to print the report for this only

#         # high_conf_view = dataset2.filter_labels(
#         #     "faster_rcnn", F("confidence") > 0.75)

#         # results = high_conf_view.evaluate_detections(
#         #     "faster_rcnn",
#         #     gt_field="ground_truth",
#         #     eval_key="eval",
#         #     compute_mAP=True,
#         # )

#         # # Get the 10 most common classes in the dataset
#         # counts = dataset2.count_values("ground_truth.detections.label")
#         # classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

#         # # Print a classification report for the top-10 classes
#         # results.print_report(classes=classes_top10)

#         # Create a dict with the source images, and a dictionary of the resilt the edge found
#         to_send = {'contents': content,
#                    'results': results_dict}

#         asyncio.get_event_loop().run_in_executor(
#             None, send_to_cloud, to_send)  # fire and forget

#         dataset2.delete()

#         return jsonify(ctime())
#     except Exception as e:
#         print(e)
#         return jsonify(ctime())


# # app.run(host='0.0.0.0', port=5000)
