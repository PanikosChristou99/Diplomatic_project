from base64 import b64decode
import io
from os import environ
import flask_cors
import flask
from time import ctime
from flask import jsonify
import fiftyone as fo
from PIL import Image
from psutil import cpu_percent
import psutil
import torch
from torchvision import models
from torch import device, cuda
from bson.json_util import loads
from multiprocessing import Process
from helper_cloud import load_dataset, predict, print_rep, print_cpu, network_monitor, setup_logger
import logging
logging.basicConfig(filename='./log/cloud.log',
                    encoding='utf-8', force=True, mode='w')

logging.info("CLOUD")

cloud_logger = setup_logger('cloud_logger', './log/cloud_logger.log')

environ['no_proxy'] = '*'

# first cpu call to start counting
psutil.Process().cpu_percent()
# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

# Get all classes the dataset contains
classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(3)

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


print_cpu("Before starting Flask :", cloud_logger)

app = flask.Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)


@app.route('/endpoint', methods=['POST'])
def hello():
    try:
        print_cpu("Cloud got content and cpu is :", cloud_logger)
        content = flask.request.get_json()

        # Create the dict of the dicts from the json sent
        content2 = loads(content)

        # Create the fiftyone dataset dict
        samples_dict = content2['samples_dict']

        dataset2 = fo.Dataset()

        # Fill the dataset with the data recieved
        for _, dict2 in samples_dict.items():
            sample = fo.Sample.from_dict(dict2)
            # print(sample)
            dataset2.add_sample(sample)

        # Store ML results here
        results_dict = {}
        sample_dict = {}

        for sample in dataset2:

            # Load image

            # image_data = b64decode(sample.data)
            # dec_data = open(sample.data, mode="r", encoding="utf-8")
            image_data = b64decode(sample.data)
            image = Image.open(io.BytesIO(image_data))

            # if model is assigned so we need to detect
            if model_name:
                print_cpu('Before ML :', cloud_logger)

                res_name = "cloud_"+environ['ML']
                detections = predict(image, device, model, classes)
                print_cpu('After ML :', cloud_logger)

                # Save predictions to dataset as the name of edge and m
                sample[res_name] = fo.Detections(
                    detections=detections)
                # Update sample with this ML res
                sample.save()
                sample_dict[sample['id']] = sample.to_dict()

                # Add them to the sent dict as well
                results_dict[sample.filepath] = dict(fo.Detections(
                    detections=detections).to_dict())

                # uncomment this to pritn report
                print_rep(dataset2, res_name)

        # the edge ran an ML so lets find its results
        if 'results_ML_name' in content2:
            print_rep(dataset2, content2['results_ML_name'])

        # with Capturing() as output:
        #     # Print a classification report for the top-10 classes
        #     results2.print_report(classes=classes_top10)
        #     # Print a classification report for the top-10 classes
        #     results1.print_report(classes=classes_top10)

        # half_of_output = int(len(output)/2)

        # out1 = "/n".join(line for line in output[:half_of_output])
        # out2 = "/n".join(line for line in output[half_of_output:])

        # dict1 = {
        #     'report': out1,
        #     'time': ctime(),
        #     'model': 'retinanet_resnet'
        # }

        # dict2 = {
        #     'report': out2,
        #     'time': ctime(),
        #     'model': 'faster_rcnn'
        # }
        # send_to_mongo(dict1)
        # send_to_mongo(dict2)

        # results.
        # to_send = {}

        # asyncio.get_event_loop().run_in_executor(
        #     None, send_to_db, content2)  # fire and forget
            print_cpu('Leavig with CPU :', cloud_logger)

        return jsonify(ctime())
    except Exception as e:
        print(e)
        return jsonify(ctime())


p = Process(target=network_monitor, args=("Cloud", cloud_logger,))
p.start()

# if 'Port' not in environ:
#     print('Did not specify "Port"')
#     exit(1)
# port = int(environ['Port'])
app.run(host='0.0.0.0', port=5000)
