
import asyncio
from helper_edge import send_to_cloud
from time import sleep

test_dict = {'heyy': ['h1', 'h2']}


asyncio.get_event_loop().run_in_executor(None, send_to_cloud, test_dict)

sleep(10)
