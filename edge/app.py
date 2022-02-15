import fiftyone as fo
from base64 import b64decode, b64encode
import flask_cors
import flask
from time import ctime
from flask import jsonify
from PIL import Image
from psutil import cpu_percent
import psutil
import torch
from torchvision import models
from torch import device, cuda
import ast
from helper_edge import load_dataset, predict, preprocess_img, print_cpu, print_rep, send_to_cloud
import asyncio
from os import environ
import io
import warnings
warnings.filterwarnings("ignore")

environ['no_proxy'] = '*'

# first cpu call to start counting
psutil.Process().cpu_percent()

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

# Get all classes the dataset contains
classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(1)


# Create model pointer and name
model = None
model_name = ""

# if we have a custom ML then use that
if 'ML' in environ:
    model_name = environ['ML']
    method = getattr(models.detection, model_name)
    model = method(pretrained=True)
    print('Using ', model_name)
    model.to(device)
    model.eval()

print_cpu('Starting  with :')

app = flask.Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)


@app.route('/endpoint', methods=['POST'])
async def hello():
    try:
        print_cpu('I got content and the cpu util is :')

        content = flask.request.get_json()

        # create the dict from the json sent
        content3 = ast.literal_eval(content)

        edge_name_str = content3['name_dict']

        edge_name = ast.literal_eval(edge_name_str)
        edge_name = edge_name['edge_name']

        content2 = content3["sample_dict"]

        dataset2 = fo.Dataset()

        # Create the fiftyone dataset dict
        for _, dict2 in content2.items():
            sample = fo.Sample.from_dict(dict2)
            # print(sample)
            dataset2.add_sample(sample)

        # Store ML results here
        results_dict = {}
        # Store the new samples from the beggining
        sample_dict = {}

        for sample in dataset2:

            # Load image

            # image_data = b64decode(sample.data)
            # dec_data = open(sample.data, mode="r", encoding="utf-8")
            image_data = b64decode(sample.data)
            image = Image.open(io.BytesIO(image_data))

            if 'Preprocessing' in environ:
                print_cpu('Before preproccesing :')

                image = preprocess_img(sample, image)
                print_cpu('After preproccesing :')

            # if model is assigned so we need to detect
            if model_name:
                print_cpu('Before ML :')
                edge_ml_name = edge_name + "_" + environ['ML']

                detections = predict(image, device, model, classes)
                print_cpu('After ML :')
                # Save predictions to dataset as the name of edge and m
                sample[edge_ml_name] = fo.Detections(
                    detections=detections)
                # Update sample with this ML res
                sample.save()
                sample_dict[sample['id']] = sample.to_dict()

                # Add them to the sent dict as well
                results_dict[sample.filepath] = dict(fo.Detections(
                    detections=detections).to_dict())

                # uncomment this to pritn report
                print_rep(dataset2, edge_ml_name)

        to_send = {'edge_name': edge_name, 'samples_dict': sample_dict}

        if 'ML' in environ:
            to_send['results_ML_name'] = edge_ml_name
            to_send['results_dict'] = results_dict

        asyncio.get_event_loop().run_in_executor(
            None, send_to_cloud, to_send)  # fire and forget

        dataset2.delete()
        print_cpu('Done with cpu at :')

        return jsonify(ctime())
    except Exception as e:
        print(e)
        return jsonify(ctime())

# if 'Port' not in environ:
#     print('Did not specify "Port"')
#     exit(1)

# port = int(environ['Port'])
app.run(host='0.0.0.0', port=5000)
