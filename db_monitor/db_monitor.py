from time import sleep
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
client = MongoClient('mongodb://mongodb:27017/')

try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
except Exception as e:
    print(e)
    exit(0)

i = 0
print('Started db_monitor')
while True:
    i += 1
    print(f'Monitor got in loop for the {i}th time')
    cursor = client.diplomatic_db.col.find()
    print('In the Database right now we have :')
    for record in cursor:
        print(record)
    sleep(50)
