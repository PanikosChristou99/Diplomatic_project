from base64 import b64decode
import io
from os import environ, getpid
import warnings
import flask_cors
import flask
from time import ctime
from flask import jsonify
import fiftyone as fo
from PIL import Image
from pandas import DataFrame, read_csv
import psutil
import torch
from torchvision import models
from torch import device, cuda
from bson.json_util import loads
from multiprocessing import Process
from helper_cloud import load_dataset, predict, print_rep, print_cpu, network_monitor, setup_logger, Capturing, send_to_mongo
import logging
from datetime import datetime
from hwcounter import Timer, count, count_end


warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

d = datetime.today()

log_name = './log/' + d.strftime('%d_%m_%H_%M') + \
    '_cloud_logger' + '.log'


logging.basicConfig(filename=log_name,
                    encoding='utf-8', force=True, filemode='w')

environ['no_proxy'] = '*'

# first cpu call to start counting
p = psutil.Process(getpid())
p.cpu_percent()

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

# Get all classes the dataset contains
classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

# Create model pointer and name
model = None
model_name = ""

cloud_csv_name_requests = './stats/' + \
    d.strftime('%d_%m_%H_%M') + '_cloud_requests_'
cloud_csv_name_monitor = './stats/' + \
    d.strftime('%d_%m_%H_%M') + '_cloud_monitor_'
cloud_reports_name = './stats/' + d.strftime('%d_%m_%H_%M') + \
    '_cloud_report_'

collumns = ['Start_CPU', 'End_CPU']


# if we have a custom ML then use that
if 'ML' in environ:
    model_name = environ['ML']
    cloud_csv_name_requests += model_name
    cloud_csv_name_monitor += model_name
    cloud_reports_name += model_name + '_'

    collumns.append('Start_ML')
    collumns.append('End_ML')

    method = getattr(models.detection, model_name)
    model = method(pretrained=True)
    print('Using ', model_name)
    model.to(device)
    model.eval()

app = flask.Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)


cloud_csv_name_requests += '.csv'
cloud_reports_name += '.txt'

df = DataFrame(columns=collumns)

df.to_csv(cloud_csv_name_requests)


@app.route('/endpoint', methods=['POST'])
def hello():
    try:
        print('I got content')

        start_cpu = count()

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

        ind = 0

        for sample in dataset2:

            ind += 1

            # Load image

            # image_data = b64decode(sample.data)
            # dec_data = open(sample.data, mode="r", encoding="utf-8")
            image_data = b64decode(sample.data)
            image = Image.open(io.BytesIO(image_data))

            ml_cpu = int(-1)

            # if model is assigned so we need to detect
            if model_name:
                if ind == 1:
                    start_ml = count()

                res_name = "cloud_"+environ['ML']
                detections = predict(image, device, model, classes)
                if ind == 1:
                    ml_cpu = int(count_end() - start_ml)

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

        output = [
            '-------------------------------------------------------------------------']

        if model_name:
            output.extend(print_rep(dataset2, res_name))

        # the edge ran an ML so lets find its results
        if 'results_ML_name' in content2:
            output.extend(print_rep(
                dataset2, content2['results_ML_name']))

        if len(output) != 1:
            output.append(
                '-------------------------------------------------------------------------')
            string = "\n".join(line for line in output)
            with open(cloud_reports_name, "a+") as f:
                f.write(string)

            models = ""
            if model_name:
                models += model_name

            # the edge ran an ML so lets find its results
            if 'results_ML_name' in content2:
                models += content2['results_ML_name']

            dict1 = {
                'time': ctime(),
                'output': string,
                'models':  models
            }

            # TODO UNCOMMENT THIS
            # send_to_mongo(dict1)

        dataset2.delete()
        end_cpu = int(count_end() - start_cpu)

        data = {'cpu_cycles': end_cpu, 'ml_cycles': ml_cpu}

        df2 = read_csv(cloud_csv_name_requests, index_col=0)
        df3 = df2.append(
            data, ignore_index=True)
        df3.to_csv(cloud_csv_name_requests, mode='w')

        return jsonify(ctime())
    except Exception as e:
        print(e)
        return jsonify(ctime())


p2 = Process(target=network_monitor, args=(
    "Cloud", cloud_csv_name_monitor))
p2.start()

# if 'Port' not in environ:
#     print('Did not specify "Port"')
#     exit(1)
# port = int(environ['Port'])
app.run(host='0.0.0.0', port=5000)
