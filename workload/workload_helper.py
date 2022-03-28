from datetime import datetime
import logging
from ast import Dict, While
from base64 import b64encode
import json
from os import path
from time import sleep
import fiftyone.zoo as foz
from pandas import read_csv
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


# Function that sends a  dictionary in json format to all the edges
def run_send_thread(workloader_csv_name, dataset, images, edge_urls):

    while True:
        try:

            for i, edge_url in enumerate(edge_urls):

                start_time = datetime.now()
                req_times = []
                for _ in range(images[-1]):

                    # print(images)
                    # print(edge_urls)
                    # print(images[-1])
                    start_req_time = datetime.now()
                    predictions_view = dataset.take(images[i])

                    # Dictionary with the picture encoded in base64, as well as all attributes the image has like what it contains, dimensions etc. (What fiftyone natively has)
                    dicts = {}
                    # This dict is the same as above but without the picture encoded, I print this some tims to debug stuff
                    dict_no_data = {}

                    for sample in predictions_view:
                        with open(sample.filepath, "rb") as image:
                            # Add to the smaller the dct the data for thsi sample
                            dict_no_data[sample['id']] = sample.to_dict()
                            # Write the base64 encoded image to the Sample
                            sample['data'] = b64encode(
                                image.read()).decode('utf-8')
                            # Add the sample to the dict to be sent as workload
                            dicts[sample['id']] = sample.to_dict()

                    # Print the images names we are sending to ensure we are sending different images each time
                    print('Sending to', edge_url)

                    print_images_names(dict_no_data)

                    contents2 = json.dumps(dicts)

                    res = requests.post(
                        edge_url, json=contents2, proxies=proxies)
                    print('response from ', edge_url, ' : ', res.text)

                    req_times.append(
                        (datetime.now() - start_req_time).microseconds/1000)

                end_time = (
                    datetime.now() - start_time).microseconds/1000

                data = {"edge_name": edge_url,
                        "total_time_milli": end_time, "req_times_milli": req_times}

                df2 = read_csv(workloader_csv_name, index_col=0)
                df2 = df2.append(
                    data, ignore_index=True)
                df2.to_csv(workloader_csv_name, mode='w')

        except Exception as e2:
            print('Exception running thread', edge_url)
            print(e2)
            sleep(2)


def print_images_names(dict_no_data: dict):
    for a, b in dict_no_data.items():
        base_path = path.basename(b['filepath'])
        print(base_path)


def print_cpu(string: str, logger: logging.Logger, p=psutil.Process(),):
    perc = p.cpu_percent()
    string2 = string + str(perc) + '%'
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
