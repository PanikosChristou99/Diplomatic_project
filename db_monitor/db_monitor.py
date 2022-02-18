from time import sleep
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from os import environ
import logging
logging.basicConfig(filename='./log/dbMonitor.log',
                    encoding='utf-8', force=True, mode='w')
client = MongoClient('mongodb://mongodb:27017/')

logging.info("DB MONITOR")

try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
except Exception as e:
    print(e)
    exit(0)

sleep_time = 60

if "Monitor_sleep" in environ:
    sleep_time = int(environ['Monitor_sleep'])

i = 0
print('Started db_monitor')
while True:
    i += 1
    print(f'Monitor got in loop for the {i}th time')

    res = list(client.diplomatic_db.col.find())

    size = len(res)
    print(f'In the Database right now we have {size} documents')
    sleep(sleep_time)
