from time import sleep
from pymongo import MongoClient
client = MongoClient('mongodb://10.16.3.96:27017/')
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
