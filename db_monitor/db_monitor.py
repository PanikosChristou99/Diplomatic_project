import json
from time import sleep
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from os import environ
import logging
logging.basicConfig(filename='./log/dbMonitor.log',
                    encoding='utf-8', force=True, filemode='w')
client = MongoClient('mongodb://mongodb:27017/')

logging.info("DB MONITOR")

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


dbMonitor_logger = setup_logger(
    'dbMonitor_logger', './log/dbMonitor_logger.log')

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
print('Started db_monitor')
while True:
    i += 1
    sleeps += 1
    print(f'Monitor got in loop for the {i}th time')

    res = list(client.diplomatic_db.col.find())

    size = len(res)
    print(f'In the Database right now we have {size} documents')

    if sleeps >= 3:
        print(json.dumps(res))
        sleeps = 0

    sleep(sleep_time)
