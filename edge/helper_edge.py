# sample of the contents it recieves from workloader without the data

# content2 = {"61829d971346ff92a0302bc9": {"filepath": "/root/fiftyone/coco-2017/validation/data/000000037777.jpg", "tags": ["validation"], "metadata": {"_cls": "ImageMetadata", "width": 352, "height": 230}, "ground_truth": {"_cls": "Detections", "detections": [{"_id": {"$oid": "61829d971346ff92a0302ad3"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "potted plant", "bounding_box": [0.2911647727272727, 0.5150869565217391, 0.02244318181818182, 0.07526086956521738], "supercategory": "furniture", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad4"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "chair", "bounding_box": [0.07528409090909091, 0.9358695652173913, 0.1747159090909091, 0.06304347826086956], "supercategory": "furniture", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad5"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "chair", "bounding_box": [0.3309659090909091, 0.8242173913043478, 0.14204545454545456, 0.1108695652173913], "supercategory": "furniture", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad6"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "dining table", "bounding_box": [0.22599431818181817, 0.7741304347826088, 0.5919318181818182, 0.21173913043478262], "supercategory": "furniture", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad7"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "refrigerator", "bounding_box": [0.8574999999999999, 0.3258260869565217, 0.1409659090909091, 0.6584347826086956], "supercategory": "appliance", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad8"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "banana", "bounding_box": [0.6268465909090909, 0.7779565217391304, 0.10750000000000001, 0.12134782608695652], "supercategory": "food", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ad9"}, "_cls": "Detection", "attributes": {}, "tags": [
# ], "label": "oven", "bounding_box": [0.39053977272727275, 0.539608695652174, 0.17096590909090909, 0.30878260869565216], "supercategory": "appliance", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ada"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "sink", "bounding_box": [0.7566193181818182, 0.5847391304347827, 0.07917613636363637, 0.015086956521739131], "supercategory": "appliance", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302adb"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "orange", "bounding_box": [0.6129829545454546, 0.8039130434782609, 0.049034090909090916, 0.06943478260869565], "supercategory": "food", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302adc"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "orange", "bounding_box": [0.6579545454545455, 0.8737391304347827, 0.04741477272727273, 0.06926086956521739], "supercategory": "food", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302add"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "orange", "bounding_box": [0.654971590909091, 0.775, 0.03332386363636364, 0.04773913043478261], "supercategory": "food", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ade"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "orange", "bounding_box": [0.5829261363636363, 0.8130869565217391, 0.042585227272727275, 0.09217391304347826], "supercategory": "food", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302adf"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "chair", "bounding_box": [0.6873579545454546, 0.7844347826086956, 0.1459375, 0.1973913043478261], "supercategory": "furniture", "iscrowd": 0}, {"_id": {"$oid": "61829d971346ff92a0302ae0"}, "_cls": "Detection", "attributes": {}, "tags": [], "label": "orange", "bounding_box": [0.6190056818181818, 0.872, 0.040170454545454544, 0.058652173913043476], "supercategory": "food", "iscrowd": 0}]}}}


import json
from time import sleep
import requests


def send_to_cloud(contents: dict):
    contents2 = json.dumps(contents)

    print('Sending to cloud what I got ')
    try:
        res = requests.post(
            'http://cloud:5001/endpoint', json=contents2)
        print('response from server:', res.text)
        # time_rec = res.text['']
        # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M:
        #  % S', time.localtime(1347517370))}')
    except Exception as e:
        print('Exception on sending to cloud')
        print(e)
