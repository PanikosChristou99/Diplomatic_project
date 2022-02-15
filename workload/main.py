from asyncio import get_event_loop
from os import environ
from time import sleep
from base64 import b64encode
import json

from joblib import PrintTime
from psutil import cpu_percent, Process

from workload_helper import load_dataset, print_images_names, run_send_thread

# By pass proxy eeror on cs dep vm
environ['no_proxy'] = '*'


# first cpu call to start counting
Process().cpu_percent()

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

if 'Edges' not in environ:
    print('You did not specfy the edges. Please use the format "Edges: edge1,edge2"')

edges_str = environ['Edges']

edges = edges_str.split(',')

num_of_images = 5

if "Images" in environ:
    num_of_images = int(environ["Images"])

sleep_time = 20
if "Sleep" in environ:
    sleep_time = int(environ["Sleep"])


i = 0

print('Starting workloader with :', cpu_percent(), '%')
while True:
    i += 1

    print(f'Workloader got in loop for the {i}th time')

    for edge in edges:

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

        edge_url = 'http://' + edge + ':5000/endpoint'

        # Send to edge the workload its workload
        get_event_loop().run_in_executor(
            None, run_send_thread, dicts, edge_url)  # fire and forget
    perc = cpu_percent()
    print(
        f'Finished the {i}th workload send with {perc} % and goint to sleep for {sleep_time}')

    sleep(sleep_time)
