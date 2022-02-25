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
from torchvision import models
from torch import device, cuda, set_num_threads
from helper_edge import load_dataset, predict, preprocess_img, print_cpu, print_rep, send_to_cloud, network_monitor, setup_logger
from os import environ, getpid
import logging
from datetime import datetime
import psutil

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

edge_name = environ['Name']
d = datetime.today()

log_name = './log/' + d.strftime('%d_%m_%H_%M') + \
    '_' + edge_name + '_logger' + '.log'

logging.basicConfig(filename=log_name, encoding='utf-8',
                    force=True, filemode='w')


# first cpu call and net work to start counting before startign threads
p = psutil.Process(getpid())
p.cpu_percent()


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

collumns = ['Start_CPU', 'End_CPU']


ml = False
# if we have a custom ML then use that
if 'ML' in environ:
    ml = True
    model_name = environ['ML']
    edge_csv_name_requests += model_name + '_'
    edge_csv_name_monitor += model_name + '_'
    collumns.append('Start_ML')
    collumns.append('End_ML')

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
    collumns.append('Start_Preprocessing')
    collumns.append('End_Preprocessing')
    collumns.append('Image_size_reduction')
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

        start_cpu = p.cpu_percent()

        content = request.get_json()

        # create the dict from the json sent
        content2 = loads(content)

        dataset2 = fo.Dataset()

        count = 0
        # Create the fiftyone dataset dict
        for _, dict2 in content2.items():
            sample = fo.Sample.from_dict(dict2)
            # print(sample)
            dataset2.add_sample(sample)
            count += 1

        # Store ML results here
        results_dict = {}
        # Store the new samples from the beggining
        sample_dict = {}

        ind = 0

        for sample in dataset2:
            ind += 1

            # Load image

            # image_data = b64decode(sample.data)
            # dec_data = open(sample.data, mode="r", encoding="utf-8")
            image_data = b64decode(sample.data)
            image = Image.open(BytesIO(image_data))

            start_pre = float(-1)
            end_pre = float(-1)
            perc_smaller = float(-1)

            if 'Preprocessing' in environ:
                if ind == 1:
                    start_pre = p.cpu_percent()

                image, perc_smaller = preprocess_img(sample, image)
                if ind == 1:
                    end_pre = p.cpu_percent()

            start_ml = float(-1)
            end_ml = float(-1)

            # if model is assigned so we need to detect
            if 'ML' in environ:
                if ind == 1:
                    start_ml = p.cpu_percent()

                edge_ml_name = edge_name + "_" + environ['ML']

                detections = predict(image, device, model, classes)

                if ind == 1:
                    end_ml = p.cpu_percent()

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
        # if model_name:
        #     print_rep(dataset2, edge_ml_name , logger=edge_logger)

        to_send = {'edge_name': edge_name, 'samples_dict': sample_dict}

        if ml:
            to_send['results_ML_name'] = edge_ml_name
            to_send['results_dict'] = results_dict

        get_event_loop().run_in_executor(
            None, send_to_cloud, to_send)  # fire and forget

        dataset2.delete()
        end_cpu = p.cpu_percent()

        data = {'Start_CPU': start_cpu, 'End_CPU': end_cpu,
                'Start_ML': start_ml, 'End_ML': end_ml,
                'Start_Preprocessing': start_pre, 'End_Preprocessing': end_pre,
                'Image_size_reduction': perc_smaller
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
    edge_name, p, edge_csv_name_monitor))

p2.start()

# if 'Port' not in environ:
#     print('Did not specify "Port"')
#     exit(1)

# port = int(environ['Pcatort'])
app.run(host='0.0.0.0', port=5000)
