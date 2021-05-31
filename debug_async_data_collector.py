
import json, requests, time, asyncio, threading, requests, sys, copy, time

from concurrent.futures import ProcessPoolExecutor
from async_data_collector import func
from scipy.interpolate import interp1d

from config import params

import aux

import pandas as pd
import numpy as np

PERIOD = params['api_call_period']
LENGTH = params['website_length']
INITIAL_TREND = 1#0
SIMULATION=False
ETOL = 4

# Define a callback for the data mining process
def between_callback():
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	loop.run_until_complete(func())
	loop.close()



# Start consuming data by calling Binance's API
t1 = threading.Thread(target=between_callback).start()

