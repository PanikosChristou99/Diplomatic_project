from asyncio import get_event_loop
from math import sqrt
from os import environ, getpid
from time import sleep
import warnings
import pandas
from psutil import Process, net_io_counters, virtual_memory
import logging
from workload_helper import load_dataset, run_send_thread
from datetime import datetime, timedelta
from hwcounter import Timer, count, count_end


warnings.simplefilter(action='ignore', category=FutureWarning)


d = datetime.now() + timedelta(hours=2)

logger_filename = './log/' + d.strftime('%m_%d_%H_%M') + '_workload_logger.log'
logging.basicConfig(filename=logger_filename,
                    encoding='utf-8', force=True,  filemode='w')


# workload_logger = setup_logger('workload_logger', './log/workload_logger.log')

# By pass proxy eeror on cs dep vm
environ['no_proxy'] = '*'

# first cpu call and net work to start counting before startign threads
p = Process(getpid())
p.cpu_percent()


# Load coco dataset so that we can get the classes of the images,
# the data is already since we had built it in the base image
dataset = load_dataset()

if 'Edges' not in environ:
    print('You did not specfy the edges. Please use the format "Edges: edge1,edge2"')
    exit(1)

edges_str = environ['Edges']

edges = edges_str.split(',')


num_of_images = [5 for _ in edges]
num_of_images.append(20)  # last one is repeated

if "Images" not in environ:
    print('You did not Images so sending 5 to all ')
else:
    images_ar = environ["Images"].split(',')
    for i, val in enumerate(images_ar):
        num_of_images[i] = int(images_ar[i])


sleep_time = 4

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])

workloader_csv_name = './stats/' + \
    d.strftime('%m_%d_%H_%M') + '_workload_monitor_'
workloader_csv_name2 = './stats/' + \
    d.strftime('%m_%d_%H_%M') + '_workload_requests_'

for i, edge in enumerate(edges):
    workloader_filename = str(edge) + \
        '_' + str(num_of_images[i]) + '_'

    workloader_csv_name += workloader_filename
    workloader_csv_name2 += workloader_filename


workloader_csv_name2 = workloader_csv_name2 + \
    '_sleepTime_' + str(sleep_time) + '.csv'

workloader_csv_name = workloader_csv_name + \
    '_sleepTime_' + str(sleep_time) + '.csv'


print('Starting workloader with :', p.cpu_percent(), '%')


df = pandas.DataFrame()

df.to_csv(workloader_csv_name)
df.to_csv(workloader_csv_name2)

edge_urls = []

for i, edge in enumerate(edges):

    edge_url = 'http://' + edge + ':5000/endpoint'
    edge_urls.append(edge_url)

get_event_loop().run_in_executor(
    None, run_send_thread, workloader_csv_name2, dataset, num_of_images, edge_urls)  # fire and forget


print('Starting workloader monitor')

bytes_sent_before = net_io_counters().bytes_sent
start_cpu = count()

sleep(1)
try:
    while True:

        bytes_sent_after = net_io_counters().bytes_sent

        diff_sent = (bytes_sent_after - bytes_sent_before) / 1000

        bytes_sent_before = net_io_counters().bytes_sent
        start_cpu = count()

        elapsed = int(count_end() - start_cpu)
        mem = virtual_memory()

        vram_used = mem.used / 1024/1024  # MBytes
        ram_used = mem.active / 1024/1024  # MBytes

        data2 = {'cpu_cycles': elapsed, 'KBytes_sent': diff_sent,
                 'vram_used_MBytes': vram_used, 'ram_active_MBytes': ram_used}

        df = df.append(data2, ignore_index=True)
        df.to_csv(workloader_csv_name, mode='w')

        sleep(sleep_time)
except Exception as e:
    print(e)
