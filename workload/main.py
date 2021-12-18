import os
import requests
from time import sleep
import fiftyone.zoo as foz
from random import randint
from base64 import b64encode
import json

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
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

    # Dump the dictionary as a string
    dictToSend = json.dumps(dicts)

    # Print the images names we are sending to ensure we are sending different images each time
    for a, b in dict_no_data.items():
        base_path = os.path.basename(b['filepath'])
        print(base_path)

    # Send to edge the workload its workload
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
