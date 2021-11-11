import os
import requests
from time import sleep
import fiftyone.zoo as foz
from random import randint
from base64 import b64encode
import json
import pprint

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="coco-2017-val",
    max_samples=500)

i = 0

while True:
    i += 1

    print(f'Workloader got in loop for the {i}th time')
    predictions_view = dataset.take(5)

    dicts = {}
    dict_no_data = {}

    for sample in predictions_view:
        with open(sample.filepath, "rb") as image:
            dict_no_data[sample['id']] = sample.to_dict()
            sample['data'] = b64encode(image.read()).decode('utf-8')
            dicts[sample['id']] = sample.to_dict()

    dictToSend = json.dumps(dicts)

    for a, b in dict_no_data.items():
        base_path = os.path.basename(b['filepath'])
        print(base_path)

    try:
        res = requests.post(
            'http://edge:5000/endpoint', json=dictToSend)
        print('response from server:', res.text)
        # time_rec = res.text['']
        # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M:
        #  % S', time.localtime(1347517370))}')
        sleep(20)
    except Exception as e:
        print('Couldn not send to edge so sleeping')
        sleep(20)
        continue
