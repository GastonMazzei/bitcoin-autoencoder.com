
from config import config, params
import json, requests, time, asyncio, threading, requests, os

if False:
	from kafkaHelper import initProducer, produceRecord
	# real time data collector
	async def async_getCryptoRealTimeData(producer, topic, crypto, time_inverval):
		# Old url
		#uri = 'https://api.coinbase.com/v2/prices/{0}-{1}/{2}'.format(crypto, params['ref_currency'], 'spot')
		# New url
		uri = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
		while True:
			t_0 = time.time()
			# call API
			res = requests.get(uri)
			if (res.status_code==200):
				# read json response
				raw_data = json.loads(res.content)
			    # add observation timestamp to json
				raw_data['data']['timestamp'] = time.time()
			    # debug / print message
				print('API request at time {0}'.format(time.time()))
			    # produce record to kafka
				produceRecord(raw_data['data'], producer, topic)
			else:
				# debug / print message
				print('Failed API request at time {0}'.format(time.time()))
			# wait
			await asyncio.sleep(time_inverval - (time.time() - t_0))
	# initialize kafka producer
	producer = initProducer()
	# define async routine
	async def func():
		await asyncio.gather(
		async_getCryptoRealTimeData(producer, config['topic_1'], params['currency_1'], params['api_call_period']),
		)
else:
	# real time data collector
	async def async_getCryptoRealTimeData(topic, crypto, time_inverval):
		# Old url
		#uri = 'https://api.coinbase.com/v2/prices/{0}-{1}/{2}'.format(crypto, params['ref_currency'], 'spot')
		# New url
		uri = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
		MAX_TRIES = 5
		while True:
			counter = 1
			t_0 = time.time()
			b = True
			# call API
			while b:
				counter += 1
				try:
					res = requests.get(uri)
					if res.status_code==200:
						b=False
				except Exception as ins:
					print('API CALL HAD A PROBLEM! INS.ARGS={ins.args}')
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
				os.system(f'echo {raw_data["data"]["timestamp"]},{raw_data["data"]["price"]} >> app_results')
			else:
			    # debug / print message
				print('Failed API request at time {0}'.format(time.time()))
			# wait
			await asyncio.sleep(time_inverval - (time.time() - t_0))
	async def func():
		await asyncio.gather(
		async_getCryptoRealTimeData(config['topic_1'], params['currency_1'], params['api_call_period']),
		)
