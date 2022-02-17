
# A function that sends to edge the dictionary
# This function is passed to a thread to be ran and be forgotten about

from io import StringIO
from multiprocessing.dummy import Process
import sys
from psutil import cpu_percent
import psutil
from pymongo import MongoClient
from base64 import b64encode
from os import environ, getcwd, path, remove
from time import sleep
import requests
from bson.json_util import dumps
import fiftyone.zoo as foz
from PIL import Image
from torchvision.transforms import functional as func
import fiftyone as fo
from fiftyone import ViewField as F

proxies = {
    "http": None,
    "https": None,
}


def load_dataset():
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name="coco-2017-val",
        max_samples=500)

    return dataset


def send_to_cloud(contents: dict):
    contents2 = dumps(contents)

    print('Sending to cloud what I got ')
    try:
        res = requests.post(
            'http://cloud:5001/endpoint', json=contents2, proxies=proxies)
        print('response from server:', res.text)
        # time_rec = res.text['']
        # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M:
        #  % S', time.localtime(1347517370))}')
    except Exception as e:
        print('Exception on sending to cloud')
        print(e)


def preprocess_img(sample, image):
    curr_path = path.abspath(getcwd())
    pic_name = path.basename(sample.filepath)
    # save a apth to modify the image on
    new_path = path.join(curr_path, pic_name)

    # image.save(new_path, quality=100, optimize=True)
    # print("Original image size = ", path.getsize(new_path) / (1024*1024))
    # remove(new_path)

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

    if 'BW' in pref_dict:
        if pref_dict['BW'] != '0':
            image.draft("L", image.size)

    image.save(new_path, quality=quality, optimize=True)
    # print("New image size = ", path.getsize(new_path) / (1024*1024))

    with open(new_path, "rb") as image:

        # Write the NEW base64 encoded image to the Sample
        sample['data'] = b64encode(image.read()).decode('utf-8')
        # Add the sample to the dict to be sent as workload

    # re open the image with the new quality
    image = Image.open(new_path)
    # remove ot not fill space
    remove(new_path)
    return image


def predict(image, device, model, classes):
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
        rel_box = [x1 / w, y1 / h,
                   (x2 - x1) / w, (y2 - y1) / h]

        detections.append(
            fo.Detection(
                label=classes[label],
                bounding_box=rel_box,
                confidence=score
            )
        )

    return detections


def print_rep(dataset2, edge_ml_name):
    # Uncomment the below to print the report for this ML
    high_conf_view = dataset2.filter_labels(
        edge_ml_name, F("confidence") > 0.75)

    results = high_conf_view.evaluate_detections(
        edge_ml_name,
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
    )

    # Get the 10 most common classes in the dataset
    counts = dataset2.count_values("ground_truth.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    print('Results for ', edge_ml_name, " are:")
    results.print_report(classes=classes_top10)


# Set up mongo connection
conn = MongoClient('mongodb://mongodb:27017/')
db = conn.diplomatic_db
collection = db.col


def send_to_mongo(contents: dict):

    # Insert Data
    rec_id1 = collection.insert_one(contents)
    print('inserted record :', rec_id1)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def print_cpu(string: str, p=psutil.Process()):
    perc = p.cpu_percent()
    print(string, perc, '%')


def network_monitor(edge_name):

    sleep_time = 60

    if "Monitor_sleep" in environ:
        sleep_time = int(environ['Monitor_sleep'])

    print(f'Starting {edge_name}')
    while True:
        bytes_sent_before = psutil.net_io_counters().bytes_sent
        bytes_recv_before = psutil.net_io_counters().bytes_recv
        sleep(sleep_time)
        diff_sent = psutil.net_io_counters().bytes_sent - bytes_sent_before
        diff_recv = psutil.net_io_counters().bytes_recv - bytes_recv_before
        print(
            f'Cloud after {sleep_time} has sent {diff_sent} and recieved {diff_recv} bytes')
