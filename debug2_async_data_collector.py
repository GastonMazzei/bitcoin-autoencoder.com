from config import config, params
import json, requests, time, asyncio, threading, requests, os
import json, requests, time, asyncio, threading, requests, sys, copy, time

from concurrent.futures import ProcessPoolExecutor
from async_data_collector import func
from scipy.interpolate import interp1d

from config import params

uri = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
MAX_TRIES = 5

print('hi!')

while True:
	counter = 1
	t_0 = time.time()
	b = True
	print('Entering while...')
	# call API
	while b:
		counter += 1
		try:
			res = requests.get(uri)
			if res.status_code==200:
				b=False
		except Exception as ins:
			print(f'API CALL HAD A PROBLEM! INS.ARGS={ins.args}')
		if counter>MAX_TRIES: 
			raise Exception(f'max retries exceeded!')
	if (res.status_code==200):
		# read json response
		raw_data = {'data':json.loads(res.content)}
		# add observation timestamp to json
		raw_data['data']['timestamp'] = time.time()
		# debug / print message
		print('API request at time {0}'.format(time.time()))
		# debug \ message in prompt
		os.system(f'echo {raw_data["data"]["timestamp"]},{raw_data["data"]["price"]} >> app_results_debug')
	else:
	    # debug / print message
		print('Failed API request at time {0}'.format(time.time()))
	# wait
	time.sleep(time_inverval - (time.time() - t_0))
