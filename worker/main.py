import os
import requests
from time import sleep, ctime
from pathlib import Path
from os import listdir
from base64 import b64encode
import random
random.seed(1)

# from PIL import Image
# os.chdir('./worker')
# print(os.getcwd())
# print(listdir(os.getcwd()))

DATA_PATH = Path('./data')

if not DATA_PATH.exists():
    raise Exception('No data folder')


def main():

    folders = listdir(DATA_PATH)
    if len(folders) == 0:
        raise Exception('No folders in {DATA_PATH}')
    # print(folders)
    string = ''

    sleep(180)

    while True:

        folder_index = random.randint(0, len(folders)-1)
        folder_name = folders[folder_index]
        folder_path = Path(os.path.join(DATA_PATH, folder_name))

        images = listdir(folder_path)
        if len(images) == 0:
            raise Exception('No images in {folder_path}')

        image_index = random.randint(0, len(images)-1)
        image_name = images[image_index]
        image_path = Path(os.path.join(folder_path, image_name))
        # img = Image.open(image_path)
        # img.show()
        with open(image_path, "rb") as image:
            string = b64encode(image.read()).decode('utf-8')

        dictToSend = {
            'time': str(ctime()),
            'photo_data': string,
            'photo_name': str(image_path)

        }

        try:
            res = requests.post(
                'http://edge:5000/tests/endpoint', json=dictToSend)
            print('response from server:', res.text)
            # time_rec = res.text['']
            # print(f'confirmation recieved at : {time.strftime(' % Y-%m-%d % H: % M: % S', time.localtime(1347517370))}')
        except Exception as e:
            sleep(20)
            continue
        sleep(20)


if __name__ == '__main__':
    main()
