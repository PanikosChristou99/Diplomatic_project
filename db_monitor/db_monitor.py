import json
from time import sleep
from pandas import DataFrame
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from os import environ
from datetime import datetime

import logging

client = MongoClient('mongodb://mongodb:27017/')


try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
except Exception as e:
    print(e)
    exit(0)

sleep_time = 60

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])


sleeps_to_print = 3

if "Sleeps_to_print" in environ:
    sleeps_to_print = int(environ['Sleeps_to_print'])

i = 0
sleeps = 0


d = datetime.today()

logger_filename = './log/' + \
    d.strftime('%d_%m_%H_%M') + '_db_monitor_logger.log'
logging.basicConfig(filename=logger_filename,
                    encoding='utf-8', force=True,  filemode='w')


workloader_csv_name = './stats/' + \
    d.strftime('%d_%m_%H_%M') + '_' + sleep_time + '_' + \
    sleeps_to_print+'_db_monitor.log'

data = {'num_of_documets': [0]}

df = DataFrame(data)

df.to_csv(workloader_csv_name)

print('Started db_monitor')
while True:
    i += 1
    sleeps += 1
    print(f'Monitor got in loop for the {i}th time')

    res = list(client.diplomatic_db.col.find())

    size = len(res)
    print(f'In the Database right now we have {size} documents')

    data2 = {'num_of_documets': size}

    df = df.append(data2, ignore_index=True)
    df.to_csv(workloader_csv_name, mode='w')

    if sleeps >= 3:
        print(json.dumps(res))
        sleeps = 0

    sleep(sleep_time)
