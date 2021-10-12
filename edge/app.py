from os import path
from pathlib import Path
from bottle import route, run, request
from time import ctime
from base64 import b64decode
# from PIL import Image
import json

from utils import convert_to_jpg, run_detector


@route('/tests/endpoint', method='POST')
def hello():
    print(request)
    input_json = request.json
    print(input_json)

    # force=True, above, is necessary if another developer
    # forgot to set the MIME type to 'application/json'
    # print('data from client:', input_json)

    # req_time = Path(input_json['time'])
    req_photo = input_json['photo_data']
    req_photo_data = b64decode(req_photo)
    req_photo_name = Path(input_json['photo_name']).absolute()

    with open(req_photo_name, "wb") as file:
        file.write(req_photo_data)

    # run_detector(req_time)

    # img = Image.open(req_time)
    # img.show()
    run_detector(convert_to_jpg(str(req_photo_name)))

    dictToReturn = {'ok': ctime()}
    return json.dumps(dictToReturn)


run(host='0.0.0.0', port=5000, debug=False)
