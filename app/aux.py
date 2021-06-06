
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


def define_x_and_y(data: np.ndarray) -> np.ndarray:
	L = len(data)
	M = 8
	result = []
	for i in range(L-M+1):
		result.append(data[i:i+M])
	return np.asarray(result)


def create_sub_model(neuralmodel='model100B'):
	m = load_model(neuralmodel)
	inputs = Input(shape=(int(m.input.shape.__str__().split(',')[1][1:]),1))
	x = m.layers[0](inputs)
	#print(len(m.layers))
	L = len(m.layers[6:-9])
	for i in range(1,6+L//2+1):
		x = m.layers[i](x)
	output = m.layers[6+L//2+1](x)
	model = Model(inputs=inputs, outputs=output, name="autoencoder")
	return model


def risk_computer(y, ypred, mc):
	m,c = mc
	ypred_linear =  np.vstack([ypred[:,0], np.polyval((m,c), ypred[:,0])]).T
	#print(f'm and c are: {m} {c}')
	latent_v = np.asarray([mse(ypred[i,:], ypred_linear[i,:]) for i in range(len(ypred))])
	return latent_v



def compute_price(v):
	WINDOW=v.shape[1]
	L = sum(v.shape)
	try:
		rem=L%WINDOW
		if rem-1>0:
			return np.concatenate(np.asarray([v[WINDOW*j,:].tolist() for j in [q for q in range(int(L/WINDOW))]]+[v[-1,-rem+1:].tolist()]))
		else:
			return np.asarray([v[WINDOW*j,:].tolist() for j in [q for q in range(int(L/WINDOW))]]).flatten()
	except:
		rem=(L-1)%WINDOW
		if rem>0:
			return np.concatenate(np.asarray([v[WINDOW*j,:].tolist() for j in [q for q in range(int(L/WINDOW)-1)]]+[v[-1,-rem:].tolist()]))
		else:
			return np.asarray([v[WINDOW*j,:].tolist() for j in [q for q in range(int(L/WINDOW)-1)]]).flatten()


def test():
	f = lambda x: compute_price(define_x_and_y(np.asarray(range(x))))==np.asarray(range(x))
	q=[f(i) for i in range(9,100)]
	t = {i+9:q[i] for i in range(len(q)) if type(q[i])==bool}
	print(f'FAILED for: {t}')
	return



def computer(ytrain, latent_v, thr=7.55e-11, sg=0, first=False, price=False, whole=False):
	OFFST = 7#8
	WINDOW=ytrain.shape[1]
	L = sum(ytrain.shape)
	latent_risk = latent_v
	if type(price)==bool and price==False:
		print('about to compute price')
		price = compute_price(ytrain)
	#print(len(price), len(latent_risk))
	if len(price)<len(latent_risk):
		latent_risk = latent_risk[:len(price)]
	else:
		price = price[:len(latent_risk)]
	L = len(price)
	assert(L == len(price) == len(latent_risk))
	if whole:
		if first:
			GAIN, trend_change_signal = produce_gain_starter(latent_risk, price, thr, sg=sg, whole=True)
		else:
			# THIS ONE IS RUN AT FIRST!
			GAIN, trend_change_signal = produce_gain(latent_risk, price, thr, sg=sg, whole=True)
	else:
		if first:
			GAIN, trend_change_signal = produce_gain_starter(latent_risk, price, thr, sg=sg)
		else:
			GAIN, trend_change_signal = produce_gain(latent_risk, price, thr, sg=sg)
	marks = [t_+OFFST for t_ in trend_change_signal]
	return GAIN, marks, price


def get_relevant(gains, marks, coord):
	origin,L = coord
	window_L = [origin+i for i in range(L)]
	relevant_indexes = [x for x in marks if x in window_L]	
	return relevant_indexes
	


def produce_gain(var,price,thr, sg=0, whole=False):
	"""
	sg=0 indicates a positive trend; c/c sg=1 please
	"""
	#print(len(var), len(price))
	assert(len(var)==len(price))
	ix = list(range(len(var)))
	trend_change_signal = np.where(var>thr, ix, np.nan)
	trend_change_signal = trend_change_signal[~np.isnan(trend_change_signal)].astype('int')
	if whole:
		return [x*(-1)**(sg+j) for j,x in enumerate(np.diff(price[[0]+trend_change_signal.tolist()]))], [0]+trend_change_signal.tolist()
	else:
		return np.sum([x*(-1)**(sg+j) for j,x in enumerate(np.diff(price[[0]+trend_change_signal.tolist()]))]), [0]+trend_change_signal.tolist()



# ex produce_gain_starter, now overwritting the previous func!
def produce_gain(var,price,thr, sg=0, whole=False):
	"""
	SCRIPT USED FOR STARTING PURPOSES: it includes the "0" index for the trading computation!
	sg=0 indicates a positive trend; c/c sg=1 please
	"""
	#print(len(var), len(price))
	assert(len(var)==len(price))
	ix = list(range(len(var)))
	trend_change_signal = np.where(var>thr, ix, np.nan)
	trend_change_signal = trend_change_signal[~np.isnan(trend_change_signal)].astype('int').tolist()
	trend_change_signal = [0]+sorted([x for x in trend_change_signal if x!=0])
	if whole:
		return [x*(-1)**(sg+j) for j,x in enumerate(np.diff(price[trend_change_signal]))], trend_change_signal
	else:
		return np.sum([x*(-1)**(sg+j) for j,x in enumerate(np.diff(price[trend_change_signal]))]), trend_change_signal



