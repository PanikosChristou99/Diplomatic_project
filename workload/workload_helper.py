from ast import Dict
import json
from os import path
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


# Function that sends a  dictionary in json format to a url
def run_send_thread(contents: dict, url: str):
    contents2 = json.dumps(contents)

    print('Sending to', url)
    try:
        res = requests.post(
            url, json=contents2, proxies=proxies)
        print('response from ', url, ' : ', res.text)
        # time_rec = res.text['']
        # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M:
        #  % S', time.localtime(1347517370))}')
    except Exception as e:
        print('Exception on sending', url)
        print(e)


def print_images_names(dict_no_data: dict):
    for a, b in dict_no_data.items():
        base_path = path.basename(b['filepath'])
        print(base_path)


def print_cpu(string: str, p=psutil.Process()):
    perc = p.cpu_percent()
    print(string, perc, '%')
