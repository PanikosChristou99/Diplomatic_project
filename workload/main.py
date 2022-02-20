from asyncio import get_event_loop
from os import environ
from time import sleep
from psutil import cpu_percent, Process, net_io_counters
import logging
from workload_helper import load_dataset, run_send_thread, setup_logger

logging.basicConfig(filename='./log/workload.log',
                    encoding='utf-8', force=True,  filemode='w')


workload_logger = setup_logger('workload_logger', './log/workload_logger.log')

# By pass proxy eeror on cs dep vm
environ['no_proxy'] = '*'


# first cpu call and net work to start counting before startign threads
Process().cpu_percent()
bytes_sent_before = net_io_counters().bytes_sent

# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

if 'Edges' not in environ:
    print('You did not specfy the edges. Please use the format "Edges: edge1,edge2"')
    exit(1)

edges_str = environ['Edges']

edges = edges_str.split(',')


num_of_images = [5 for _ in edges]

if "Images" not in environ:
    print('You did not Images so sending 5 to all ')
else:
    images_ar = environ["Images"].split(',')
    for i, val in enumerate(images_ar):
        num_of_images[i] = int(images_ar[i])

num_of_sleeps = [40 for _ in edges]

if "Sleep" not in environ:
    print('You did not specify sleep so 40 to all ')
else:
    sleep_ar = environ["Sleep"].split(',')
    for i, val in enumerate(sleep_ar):
        num_of_sleeps[i] = int(sleep_ar[i])


print('Starting workloader with :', cpu_percent(), '%')

for i, edge in enumerate(edges):

    edge_url = 'http://' + edge + ':5000/endpoint'

    time_sleep = num_of_sleeps[i]
    images = num_of_images[i]
    # Send to edge the workload its workload
    get_event_loop().run_in_executor(
        None, run_send_thread, edge_url, time_sleep, dataset, images)  # fire and forget

i = 0


sleep_time = 60

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])


while True:
    i += 1
    perc = cpu_percent()

    bytes_sent_after = net_io_counters().bytes_sent
    diff_sent = (bytes_sent_after - bytes_sent_before) / 1000
    bytes_sent_before = net_io_counters().bytes_sent

    workload_logger.info(
        f'Edge after {sleep_time} has {perc} cpu percentage and has sent {diff_sent} KBytes')

    sleep(sleep_time)
