#!/bin/python3

import json, requests, time, asyncio, threading, requests, sys, copy, time

from concurrent.futures import ProcessPoolExecutor
from async_data_collector import func
from scipy.interpolate import interp1d

from config import params

import aux

import pandas as pd
import numpy as np


# Flask or Quart?
if False:
	app = Quart(__name__)
	from quart import Quart, jsonify, render_template
else:
	from flask import Flask, jsonify, render_template
	app = Flask(__name__)



	
# Define global variables
global COUNTER
global DATA
global current
current = []


# Function for consuming the database and producing results
def producer(L=10, TAU=30, INITIAL_TREND=0, SIMULATION=False, ETOL=0):
	"""
	Consume the database and produce results
	"""

	# Set the website's display length
	ixs = [-L+1+i for i in range(L)]

	# Load models
	model = aux.create_sub_model()
	m = aux.load_model('model100B')

	# Define weights and strategy thresholds
	weights_and_thresholds = ((0.15711779144578095, 1.8359880417955482e-08), 
				  (0.24228292272871077, 2.5453470579438286e-08),
				 (0.08502584961642527, 3.800366855744631e-08),
				 (0.1107703562789264, 5.1645188098759393e-08),
				 (0.23666708871547587, 5.2190848880411914e-08),
				 (0.1681359706692673, 5.65561351336321e-08))

	# Define a line in latent space ;-)
	pol = (1.8335481401209706, -0.021849175342203025)

	
	# Auxiliary function 
	def compute_tmin_num(x,t0,TAU):
		local = [t0-i*TAU for i in range(len(x)) if t0-i*TAU>=x.min()]
		return min(local), len(local)


	# Define variables for iteration
	TOTALGAINS = [0 for _ in range(7)]
	COUNTER = 0

	# Define the display
	SHOW = [True, False][1]
	if SHOW:
		import matplotlib.pyplot as plt


	# Load initial data
	if SIMULATION:
		INITIAL_MAX = int(2*L)
		INITIAL_OFFSET = 50
		price_base = pd.read_csv('app_results', header=None).to_numpy()[INITIAL_OFFSET:INITIAL_MAX+INITIAL_OFFSET]
	else:
		price_base = pd.read_csv('app_results', header=None).to_numpy()
	MAX_BTC_FACTOR = 150000*np.max(price_base[:,1])/64800.96313898978  # 87784 worked recently (1 June 21)
	px=price_base[:,0]
	py=price_base[:,1]


	while True:

		if COUNTER==0:
			# call present time when the pipeline is ready for production; 
			# else replace with px.max() for the time being
			t0 = time.time()-ETOL if True else px.max()
			original_time = time.time()

			# interpolate the actual value with cubic splines and 
			# replace the price with this data
			f = interp1d(px, py, kind='cubic')

			# OLD definition of time
			if False:
				tmin, num = compute_tmin_num(px,t0,TAU)
				basetime = np.linspace(tmin, t0, num=num, endpoint=True)
			# NEW definition of time
			else:								
				if SIMULATION:
					# assert the database has the periodicity correctly set
					assert(int(np.mean(np.diff(px)))==params['api_call_period']==TAU)
					basetime = np.linspace(t0-(2*L-5)*TAU, t0, num=2*L-5, endpoint=True)
					price=f(basetime)
					print(basetime.shape, price.shape)
					x_for_slope, y_for_slope = basetime[:-L].copy(), price[:-L].copy()
					basetime, price = basetime[-L:], price[-L:]			
				else:
					basetime = np.linspace(t0-(L-1)*TAU, t0, num=L, endpoint=True)
					print(f'dataset max is: {px.max()}, required time is: {t0}. t0 is bigger? t0-max={t0-px.max()}')
					price=f(basetime)
					print(basetime.shape, price.shape)


			if SIMULATION:
				# Load data and define the strategy's initial sign; 0 is long and 1 is short
				p = np.polyfit(x_for_slope, y_for_slope, 1)
				import matplotlib.pyplot as plt
				plt.plot(x_for_slope, np.polyval(p,x_for_slope), '-', x_for_slope, y_for_slope, ':')
				plt.legend(['fit', 'data'])
				plt.show()
				INITIAL_SG = 0 if p[0]>0 else 1
				print('INITIAL SG IS: ',INITIAL_SG)
				f, ax = plt.subplots()
				ax.plot(basetime, np.polyval(p, basetime))
				ax.plot(basetime, price)
				f.savefig('initial_fit')
			else: 
				INITIAL_SG = INITIAL_TREND
			sg = {i:INITIAL_SG for i in range(6)}

		
		else:
			if True:
				# compensating for the accumulated time error in 8 calls
				if SIMULATION:
					tnow = np.max(px_experiment)
				else:
					tnow = time.time()-ETOL if True else t0+TAU*COUNTER
				try:
					basetime = np.linspace(tnow-7*TAU, tnow, num=8, endpoint=True)
					p = np.polyfit(px_experiment, py_experiment, 1)
					f = interp1d(px_experiment, py_experiment, kind='cubic')
					price = f(basetime)
					print(f'Deviation report: computing data from {tnow} ; ideally we should be at: {(original_time+TAU*COUNTER)} (lap {COUNTER})\n')
					if SIMULATION:
						plt.plot(basetime, price, '-', px_experiment, py_experiment, ':');plt.show()
				except:
					tnownew = px_experiment.max()
					basetime = np.linspace(tnownew-7*TAU, tnownew, num=8, endpoint=True)
					p = np.polyfit(px_experiment, py_experiment, 1)
					f = interp1d(px_experiment, py_experiment, kind='cubic')
					price = f(basetime)
					print(f'WARNING: time-tolerance failed; computing from max: {tnownew}, previous was {tnow} - L*WINDOW seconds ; ideally we should be at: {(original_time+TAU*COUNTER)} (lap {COUNTER})\n')
					if SIMULATION:
						plt.plot(basetime, price, '-', px_experiment, py_experiment, ':');plt.show()	
			else:
				# the accumulated time error in 8 calls can be neglected
				price=py


		# Compute the (N,8) shaped strips and normalize them
		if abs(150000*np.max(price)/64800.96313898978-MAX_BTC_FACTOR)>MAX_BTC_FACTOR*0.05:
			print(f'\n\n\nUPDATING BTC FACTOR! {abs(150000*np.max(price)/64800.96313898978-MAX_BTC_FACTOR)/MAX_BTC_FACTOR}\n\n\n')
			MAX_BTC_FACTOR = 150000*np.max(price)/64800.96313898978 
		price = price/MAX_BTC_FACTOR
		y = aux.define_x_and_y(price)
		print('THE SHAPE IS: ',y.shape)

		# Predict over data
		ypred = model.predict(y.reshape(y.shape[0],y.shape[1],1))
		latent_v = aux.risk_computer(y, ypred, pol)

		# Compute the prediction of the price
		predicted_price = aux.compute_price(m.predict(y.reshape(y.shape[0],y.shape[1],1)))
		assert(len(predicted_price)==y.shape[0]+y.shape[1]-1)

		# Rescale the predicted price into the real price
		predicted_price = (MAX_BTC_FACTOR*predicted_price).tolist()
		price *= MAX_BTC_FACTOR


		# Compute the marks and gains for each strategy
		if COUNTER==0:
			total_gains, total_marks, last_price = [],[],[]
			for IX in range(6):
				local_gain, local_marks, _ = aux.computer(y, latent_v,
									 thr=weights_and_thresholds[IX][1],
									 sg=sg[IX], first=False, price=price, whole=True)
				total_gains.append(local_gain.copy())
				total_marks.append(local_marks.copy())
				last_price.append(price[int(local_marks[-1])])

			# Update the trading direction
			sg = {k:sg[k]+len(total_marks[k])-1 for k in sg.keys()}

		else:

			total_gains, total_marks = [],[]
			for IX in range(6):
				if latent_v[-1]>weights_and_thresholds[IX][1]:
					sg[IX] += 1
				if sg[IX]%2==0:
					result = price[-1] - last_price[IX]
				else:
					result = last_price[IX] - price[-1]
				last_price[IX] = price[-1]
				total_gains.append([result])


		# Compute the weighted portfolio results
		portfolio_results = ([sum([sum(total_gains[i])*x[0] for i,x in enumerate(weights_and_thresholds)])] + [sum(x) for x in total_gains]) 

		# Append the gains to the total gains
		for _ in range(7):
			TOTALGAINS[_] += portfolio_results[_]

		# Display the portfolio results
		print(f'local portfolio results are: {portfolio_results}'\
			f'\nglobal portfolio results are" {TOTALGAINS}')


		# Update the displayed results
		if COUNTER==0:
			displayed_price = price.tolist().copy()
			displayed_predicted_price = predicted_price.copy()
		else:

			displayed_price = (displayed_price[1:] + price[-1:].tolist()).copy()
			displayed_predicted_price = (displayed_predicted_price[1:] + predicted_price[-1:]).copy() 


		# Yield the results
		yield ([[ixs, displayed_price, displayed_predicted_price],
			 [[round(x[0],2) if sg[i]%2==0 else 0 for i,x in enumerate(weights_and_thresholds)],
			[0 if sg[i]%2==0 else round(x[0],2) for i,x in enumerate(weights_and_thresholds)]],
			 TOTALGAINS])


		# for debugging purposes		
		if SHOW:
			if COUNTER==0:
				import matplotlib.pyplot as plt
				#print(f'Sg is now: {sg}')
				plt.plot(ixs, py[-L:], 'o', ixs, price, '-', ixs, predicted_price, ':')
				plt.legend(['data', 'fixed-T interpol', 'AE prediction'])
				plt.show()
			else:
				#print(f'Sg is now: {sg}')
				plt.plot(ixs, displayed_price, 'o', ixs, displayed_predicted_price, ':')
				plt.legend(['data',  'AE prediction'])
				plt.show()

		# increase counter and update data
		COUNTER+=1
		if SIMULATION:
			price_base_extra = pd.read_csv('app_results',
						header=None).to_numpy()[INITIAL_OFFSET+COUNTER+INITIAL_MAX+1-8:INITIAL_OFFSET+COUNTER+INITIAL_MAX+1]
			price_base_extra_experiment = pd.read_csv('app_results',
							 header=None).to_numpy()[INITIAL_OFFSET+COUNTER+INITIAL_MAX+1-16:INITIAL_OFFSET+COUNTER+INITIAL_MAX+1]
		else:
			price_base_extra = pd.read_csv('app_results', header=None).to_numpy()[-8:]
			price_base_extra_experiment = pd.read_csv('app_results', header=None).to_numpy()[-20:]
		px_experiment=price_base_extra_experiment[:,0]
		py_experiment=price_base_extra_experiment[:,1]
		px=price_base_extra[:,0]
		py=price_base_extra[:,1]




# Define consuming period and website's length
# Please set manually the initial sign of the strategy
# and the parameter Simulation to False.
# the initial sign can be checked at: https://www.binance.com/en/trade/BTC_USDT?type=spot
# and please finally check the HTML file to be set up with the same periodicity as the API
# calls. INITIAL_TREND 1 means short and 0 means long.
# ETOL is the tolerance to replace present time with a given delay, in seconds.
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


# Define web application functions
@app.route('/')
def index():
	return render_template('index.html')
def run_app():
	app.run(debug=False)


# Start consuming data by calling Binance's API
if not SIMULATION:
	t1 = threading.Thread(target=between_callback).start()


# Wait until "LENGTH" samples are gathered; 10% extra time
# to handle bugs
if not SIMULATION:
	pass
#	time.sleep(int(PERIOD*LENGTH*1.1))

# Start the website in another thread
if not SIMULATION:
	t2 = threading.Thread(target=run_app).start()

# Instantiate the data for the website and update it with the same frequency 
# used for the data mining process
# wait half a second for database tolerance!
time.sleep(ETOL-2.5)
DATA = producer(LENGTH, PERIOD, INITIAL_TREND, SIMULATION, ETOL)

# First call
current = DATA.__next__()
try:
	time.sleep(PERIOD - (time.time() - t_0))
except:
	# more than "P" seconds were 
	# spent retrieving the __next__
	pass


# Define a web-application function that consumes
# previously-defined data
@app.route('/data')
def data():
	global current
	return jsonify({"data":current})


# Start the periodic infinite call
while True:
	t_0 = time.time()
	current = DATA.__next__()
	try:
		time.sleep(PERIOD - (time.time() - t_0))
	except:
		# more than "P" seconds were 
		# spent retrieving the __next__
		pass








