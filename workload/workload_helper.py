import logging
from ast import Dict, While
from base64 import b64encode
import json
from os import path
from time import sleep
import fiftyone.zoo as foz
import psutil
import requests

proxies = {
    "http": None,
    "https": None,
}


# I help clearing out the main file


def load_dataset():
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name="coco-2017-val",
        max_samples=500)

    return dataset


# Function that sends a  dictionary in json format to a url in a time frame
def run_send_thread(edge_url: str, time_sleep: int, dataset, num_of_images: int):

    while True:
        try:
            predictions_view = dataset.take(num_of_images)

            # Dictionary with the picture encoded in base64, as well as all attributes the image has like what it contains, dimensions etc. (What fiftyone natively has)
            dicts = {}
            # This dict is the same as above but without the picture encoded, I print this some tims to debug stuff
            dict_no_data = {}

            for sample in predictions_view:
                with open(sample.filepath, "rb") as image:
                    # Add to the smaller the dct the data for thsi sample
                    dict_no_data[sample['id']] = sample.to_dict()
                    # Write the base64 encoded image to the Sample
                    sample['data'] = b64encode(image.read()).decode('utf-8')
                    # Add the sample to the dict to be sent as workload
                    dicts[sample['id']] = sample.to_dict()

            # Print the images names we are sending to ensure we are sending different images each time
            print_images_names(dict_no_data)

            contents2 = json.dumps(dicts)

            print('Sending to', edge_url)
            try:
                res = requests.post(
                    edge_url, json=contents2, proxies=proxies)
                print('response from ', edge_url, ' : ', res.text)
                # time_rec = res.text['']
                # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M:
                #  % S', time.localtime(1347517370))}')
            except Exception as e:
                print('Exception on sending', edge_url)
                print(e)
        except Exception as e2:
            print('Exception running thread', edge_url)
            print(e2)
        sleep(time_sleep)


def print_images_names(dict_no_data: dict):
    for a, b in dict_no_data.items():
        base_path = path.basename(b['filepath'])
        print(base_path)


def print_cpu(string: str, logger: logging.Logger, p=psutil.Process(),):
    perc = p.cpu_percent()
    string2 = string + perc + '%'
    logger.info(string2)


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
