from asyncio import get_event_loop
from math import sqrt
from os import environ, getpid
from time import sleep
import warnings
import pandas
from psutil import Process, net_io_counters
import logging
from workload_helper import load_dataset, run_send_thread
from datetime import datetime


from hwcounter import Timer, count, count_end
from time import sleep
from math import sqrt

start = count()
sqrt(144) / 12
elapsed = count_end() - start
print(f'elapsed cycles: {elapsed}')


warnings.simplefilter(action='ignore', category=FutureWarning)


d = datetime.today()

logger_filename = './log/' + d.strftime('%d_%m_%H_%M') + '_workload_logger.log'
logging.basicConfig(filename=logger_filename,
                    encoding='utf-8', force=True,  filemode='w')


# workload_logger = setup_logger('workload_logger', './log/workload_logger.log')

# By pass proxy eeror on cs dep vm
environ['no_proxy'] = '*'

# first cpu call and net work to start counting before startign threads
p = Process(getpid())
p.cpu_percent()

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


sleep_time = 60

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])

workloader_csv_name = './stats/' + d.strftime('%d_%m_%H_%M') + '_w_'

for i, edge in enumerate(edges):
    workloader_filename = str(edge) + \
        '_' + str(num_of_images[i]) + '_' + str(num_of_sleeps[i]) + '_'

    workloader_csv_name += workloader_filename


workloader_csv_name = workloader_csv_name + \
    '_sleepTime_' + str(sleep_time) + '.csv'


print('Starting workloader with :', p.cpu_percent(), '%')

data = {'CPU_Perc': [p.cpu_percent()], 'KBytes_sent': [0], }

df = pandas.DataFrame(data)

df.to_csv(workloader_csv_name)


for i, edge in enumerate(edges):

    edge_url = 'http://' + edge + ':5000/endpoint'

    time_sleep = num_of_sleeps[i]
    images = num_of_images[i]
    # Send to edge the workload its workload
    get_event_loop().run_in_executor(
        None, run_send_thread, edge_url, time_sleep, dataset, images)  # fire and forget

print('Starting workloader monitor')
i = 0
try:
    while True:
        i += 1

        bytes_sent_after = net_io_counters().bytes_sent
        diff_sent = (bytes_sent_after - bytes_sent_before) / 1000
        bytes_sent_before = net_io_counters().bytes_sent

        data2 = {'CPU_Perc': p.cpu_percent(), 'KBytes_sent': diff_sent}

        df = df.append(data2, ignore_index=True)
        df.to_csv(workloader_csv_name, mode='w')

        sleep(sleep_time)
except Exception as e:
    print(e)
