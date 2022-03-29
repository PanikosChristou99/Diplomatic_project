from datetime import datetime, timedelta
from os import environ
import sys
from time import sleep
from unicodedata import name
import dotenv
import subprocess
import copy

# Ti theloume
#
# DIffrent ML - Ola idia
# idio ML alla precpr kai pos antepskelthe to cloud
# disable on not disable preproccesing
# arithmos eikonon ola idia
# Sleep time sixnotita dipalsia tou edge1
# Tets image reduction vs accuarcy
# How quality effects accuracy kai metrcis


def write_new_env(env: dict):
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)

    for key, val in env.items():
        environ[key] = str(val)
        # Write changes to .env file.
        dotenv.set_key(dotenv_file, key, str(val))


def update_env(key: str, val: str):
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)

    environ[key] = val
    # Write changes to .env file.
    dotenv.set_key(dotenv_file, key, val)


def write_to_done_file(msg: str):
    msg += '\n'
    with open(done_file, 'a') as f:
        f.write(msg)


def run_compose(sleep_time: int):

    d = datetime.now()
    env_name = './volume/runs/env/' + d.strftime('%H_%M_%d_%m') + '_env.txt'
    output_name = './volume/runs/output/' + \
        d.strftime('%H_%M_%d_%m') + '_output.txt'

    # copy .env
    with open(env_name, 'w') as file_1, open('./.env', 'r') as file_2:
        for line in file_2:
            file_1.write(line)

    with open('./temp.sh', 'wb') as f:
        f.write(str.encode('# !/bin/bash\n'))
        f.write(str.encode(
            'docker-compose -f "./docker-compose.yml" up --build -d &\n'))
        f.write(str.encode(f'echo "Sleeping for {sleep_time} secs"\n'))
        f.write(str.encode('sleep ' + str(sleep_time) + 's\n'))
        f.write(str.encode('docker-compose stop\n'))

    with open(output_name, "a") as output:
        process = subprocess.Popen('./temp.sh', shell=True, stdout=output)
        process.wait()

    write_to_done_file("Pog")


done_file = "./proggress.txt"
models = ["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
          "fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "maskrcnn_resnet50_fpn"]
prepproccessing_parameters = {
    'BW': [0, 1],
    "resize": ['25%', '75%'],
    "resize": ['25%', '50%', "75%"], "quality": ['25%', '50%', "75%"]}
default_env = {
    'cloud_ml': 'fasterrcnn_mobilenet_v3_large_320_fpn',
    'sleep_cloud': 4,
    'edge1_ml': 'fasterrcnn_mobilenet_v3_large_fpn',
    'edge2_ml': 'fasterrcnn_resnet50_fpn',
    'edge1_pre': 'BW,1,resize,25%,quality,25%',
    'edge2_pre': '',
    'edge1_sleep': 4,
    'edge2_sleep': 4,
    'workload_edges': 'edge1,edge2',
    'workload_num_of_images': '1,1,5',
    'workload_sleep': 4,
    'test_var': 'newvalue'

}

open(done_file, 'w').close()


def step_one(secs: int):

    for bw in ["0"]:
        for pre in ["", "25%", "50%", "75%", "99%"]:

            for model1 in models:
                other = [x for x in models if x != model1]
                for model2 in other:
                    write_new_env(default_env)
                    if pre == "":
                        str1 = f'BW,{bw}'
                        str2 = f'BW,{bw}'
                        update_env('edge1_pre', str1)
                        update_env('edge2_pre', str2)
                    else:
                        str1 = f'BW,{bw},resize,{pre},quality,{pre}'
                        str2 = f'BW,{bw},resize,{pre},quality,{pre}'
                        update_env(
                            'edge1_pre', str1)
                        update_env(
                            'edge2_pre', str2)

                    update_env('edge1_ml', model1)
                    update_env('edge2_ml', model2)

                    curr_time = datetime.now().strftime('%H_%M_%d_%m')
                    string = f'{pre},{bw},{model1},{model2},{curr_time}'
                    write_to_done_file(string)
                    run_compose(secs)

                    curr_time = datetime.now().strftime('%H_%M_%d_%m')
                    string = f'DONE,{curr_time}'
                    write_to_done_file(string)


# def step_two(secs):

#     # Two
#     # Edges have same ML and cloud has fasterrcnn_mobilenet_v3_large_320_fpn
#     # but each has a different prepprocessing 0-25 / 50-75
#     # also play with BW
#     # Outcome see hwo prepprocessign affects metrics

#     write_new_env(default_env)
#     update_env('cloud_ml', 'fasterrcnn_mobilenet_v3_large_320_fpn')
#     update_env('edge1_ml', 'fasterrcnn_mobilenet_v3_large_fpn')
#     update_env('edge2_ml', 'fasterrcnn_mobilenet_v3_large_fpn')

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Starting step two at' + curr_time
#     write_to_done_file(string)

#     for bw in ["0", "1"]:
#         string1 = f'BW,{bw}'
#         string2 = f'BW,{bw},resize,25%,quality,25%'

#         update_env('edge1_pre', string1)
#         update_env('edge2_pre', string2)

#         run_compose(secs)

#     for bw in ["0", "1"]:
#         string1 = f'BW,{bw},resize,50%,quality,50%'
#         string2 = f'BW,{bw},resize,75%,quality,75%'

#         update_env('edge1_pre', string1)
#         update_env('edge2_pre', string2)

#         run_compose(secs)

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Done with step two at' + curr_time
#     write_to_done_file(string)


# def step_three(secs):

#     # Three
#     # Different ML for cloud with different 2 edges 0 percent and 75 percent prepro with no ML
#     # Outcome see how each algo is affected by big prepproccesing

#     write_new_env(default_env)
#     update_env('edge1_ml', '')
#     update_env('edge2_ml', '')
#     update_env('edge1_pre', 'BW,0')
#     update_env('edge2_pre', 'BW,0,resize,85%,quality,85%')

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Starting step three at' + curr_time
#     write_to_done_file(string)

#     for model1 in models:

#         update_env('cloud_ml', model1)

#         run_compose(secs)

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Done with step three at' + curr_time
#     write_to_done_file(string)


# def step_four(secs):

#     # Four
#     # All have all MLs and
#     # See how num of images effects stats One has 50% preproccessing

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Starting step four at' + curr_time
#     write_to_done_file(string)

#     write_new_env(default_env)
#     update_env('edge1_pre', 'BW,1,resize,50%,quality,50%')
#     update_env('edge2_pre', '')

#     for num in ["5,5", "10,10", "20,20", "40,40"]:
#         update_env('workload_num_of_images', num)

#         for model in models:
#             update_env('edge1_ml', model)
#             update_env('edge2_ml', model)

#             run_compose(secs)

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Done with step four at' + curr_time
#     write_to_done_file(string)


# def step_five(secs):

#     # Five
#     # Same edges but reverse quality vs resize
#     # Px 0 -75 je 75-0 na do accuraccy je size
#     # See all metrics

#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Starting step five at' + curr_time
#     write_to_done_file(string)

#     write_new_env(default_env)
#     update_env('edge1_pre', 'BW,0,resize,80%,quality,0%')
#     update_env('edge2_pre', 'BW,0,resize,0%,quality,80%')

#     for model in models:
#         update_env('edge1_ml', model)
#         update_env('edge2_ml', model)
#         run_compose(secs)
#     curr_time = datetime.now().strftime('%H_%M_%d_%m')
#     string = 'Done with step five at' + curr_time
#     write_to_done_file(string)
if __name__ == '__main__':

    secs = 60 * 10 * 2  # 6 mins
    step_one(secs)
    # step_two(secs)
    # step_three(secs)
    # step_four(secs)
    # step_five(secs)
