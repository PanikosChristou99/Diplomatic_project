from asyncio import get_event_loop
from io import BytesIO
import warnings
from json import loads
from multiprocessing import Process
import fiftyone as fo
from base64 import b64decode
import flask_cors
from time import ctime
from flask import Flask, jsonify, request
from PIL import Image
from pandas import DataFrame, read_csv
from psutil import cpu_count
from torchvision import models
from torch import device, cuda, set_num_threads
from helper_edge import load_dataset, predict, preprocess_img, print_cpu, print_rep, send_to_cloud, network_monitor, setup_logger
from os import environ, getpid
import logging
from datetime import datetime
import psutil
from hwcounter import Timer, count, count_end

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


edge_name = environ['Name']
d = datetime.today()

log_name = './log/' + d.strftime('%d_%m_%H_%M') + \
    '_' + edge_name + '_logger' + '.log'

logging.basicConfig(filename=log_name, encoding='utf-8',
                    force=True, filemode='w')


environ['no_proxy'] = '*'

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

# Get all classes the dataset contains
classes = dataset.default_classes

device = device("cuda:0" if cuda.is_available() else "cpu")

set_num_threads(1)


# Create model pointer and name
model = None
model_name = ""

edge_csv_name_requests = './stats/' + \
    d.strftime('%d_%m_%H_%M') + '_' + edge_name+'_requests_'
edge_csv_name_monitor = './stats/' + \
    d.strftime('%d_%m_%H_%M') + '_'+edge_name+'_monitor_'

collumns = ['cpu_cycles', 'milli_taken']


ml = False
# if we have a custom ML then use that
if 'ML' in environ:
    ml = True
    model_name = environ['ML']
    edge_csv_name_requests += model_name + '_'
    edge_csv_name_monitor += model_name + '_'
    collumns.append('ml_cycles')

    method = getattr(models.detection, model_name)
    model = method(pretrained=True)
    print('Using ', model_name)
    model.to(device)
    model.eval()

pre = False

if 'Preprocessing' in environ:
    pre = True
    preferences_str = environ['Preprocessing']
    preferences_str_replaced = preferences_str.replace(',', '_')
    edge_csv_name_requests += preferences_str_replaced+''
    edge_csv_name_monitor += preferences_str_replaced+''
    collumns.append('pre_cycles')
    collumns.append('image_size_reduction')
# print_cpu('Starting  with :', edge_logger)


if 'Name' not in environ:
    print('You did not specify edge name, please add "Name: EdgeX" at the compose file ')

app = Flask(__name__)
# This allows for running the app and taking in requests from the same computer
flask_cors.CORS(app)

edge_csv_name_requests += '.csv'

df = DataFrame(columns=collumns)

df.to_csv(edge_csv_name_requests)


@app.route('/endpoint', methods=['POST'])
async def hello():
    try:

        print('I got content')

        start_time = datetime.now()
        start_cpu = count()
        ml_cpu_temp = -1
        pre_cpu_temp = -1

        content = request.get_json()

        # create the dict from the json sent
        content2 = loads(content)

        dataset2 = fo.Dataset()

        count2 = 0
        # Create the fiftyone dataset dict
        for _, dict2 in content2.items():
            sample = fo.Sample.from_dict(dict2)
            # print(sample)
            dataset2.add_sample(sample)
            count2 += 1

        # Store ML results here
        results_dict = {}
        # Store the new samples from the beggining
        sample_dict = {}

        ind = 0

        pre_cpu = -1
        perc_smaller = -1
        ml_cpu = -1

        for sample in dataset2:
            ind += 1

            # Load image

            # image_data = b64decode(sample.data)
            # dec_data = open(sample.data, mode="r", encoding="utf-8")
            image_data = b64decode(sample.data)
            image = Image.open(BytesIO(image_data))

            if 'Preprocessing' in environ:
                if ind == 1:
                    pre_cpu_temp = count_end()
                    start_pre = count()

                image, perc_smaller = preprocess_img(sample, image)
                if ind == 1:
                    pre_cpu = count_end() - start_pre
                    start_cpu = count()

            # if model is assigned so we need to detect
            if 'ML' in environ:
                if ind == 1:
                    ml_cpu_temp = count_end()
                    start_ml = count()

                edge_ml_name = edge_name + "_" + environ['ML']

                detections = predict(image, device, model, classes)

                if ind == 1:
                    ml_cpu = count_end() - start_ml
                    start_cpu = count()

                # Save predictions to dataset as the name of edge and m
                sample[edge_ml_name] = fo.Detections(
                    detections=detections)
                # Update sample with this ML res
                sample.save()
                sample_dict[sample['id']] = sample.to_dict()

                # Add them to the sent dict as well
                results_dict[sample.filepath] = dict(fo.Detections(
                    detections=detections).to_dict())

        # uncomment this to pritn report IT HAS BUGS
        # if model_name:
        #     print_rep(dataset2, edge_ml_name , logger=edge_logger)

        to_send = {'edge_name': edge_name, 'samples_dict': sample_dict}

        if ml:
            to_send['results_ML_name'] = edge_ml_name
            to_send['results_dict'] = results_dict

        get_event_loop().run_in_executor(
            None, send_to_cloud, to_send)  # fire and forget

        num_of_images = len(dataset2)

        dataset2.delete()

        end_cpu = -1
        if 'ML' in environ and 'Preprocessing' in environ:
            end_cpu = count_end() - start_cpu + pre_cpu_temp + ml_cpu_temp
        elif 'ML' in environ:
            end_cpu = count_end() - start_cpu + ml_cpu_temp

        elif 'Preprocessing' in environ:
            end_cpu = count_end() - start_cpu + pre_cpu_temp
        else:
            end_cpu = count_end() - start_cpu

        time_taken = datetime.now() - start_time

        data = {'cpu_cycles': end_cpu, 'ml_cycles': ml_cpu, 'pre_cycles': pre_cpu,
                'image_size_reduction': perc_smaller,  'milli_taken': (time_taken.microseconds / 1000), 'num_of_images': num_of_images
                }

        df2 = read_csv(edge_csv_name_requests, index_col=0)
        df2 = df2.append(
            data, ignore_index=True)
        df2.to_csv(edge_csv_name_requests, mode='w')

        return jsonify(ctime())
    except Exception as e:
        print(e)
        return jsonify(ctime())


sleep_time = 60

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])


edge_csv_name_monitor += '_sleepTime_' + \
    str(sleep_time) + '.csv'

p2 = Process(target=network_monitor, args=(
    edge_name, edge_csv_name_monitor))

p2.start()

# if 'Port' not in environ:
#     print('Did not specify "Port"')
#     exit(1)

# port = int(environ['Pcatort'])
app.run(host='0.0.0.0', port=5000)
