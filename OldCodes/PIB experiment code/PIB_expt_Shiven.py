import numpy as np
import pylab as pl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from keras import backend as K
import scipy.integrate as spi
import scipy as sp
import scipy.optimize as spo
import scipy.io as spio

dat = spio.loadmat('MEA-37063-stim.mat')
dat = dat['data'][0][0]
Ts = dat[13]
Cs = dat[14]
#StimTimes = dat[17]
StimTimes = Ts[Cs==60]

# turn into binary vectors synced up to the causal state
dt = 0.1 # fix this
Ts = Ts/16000 # sample rate
StimTimes = StimTimes/16000
xs = []
rs = []
ys = []
for i in range(int(np.max(Ts)/dt)):
	# record silence as well
	# identify the causal state
	mask = StimTimes<i*dt
	foo = StimTimes[mask]
	if np.sum(foo)>0:
		foo = np.max(foo)
		xs.append(min(100,int(i-(foo/dt))))
		mask = StimTimes>=(i+1)*dt
		foo = StimTimes[mask]
		foo = np.min(foo)
		ys.append(min(100,int((foo/dt)-i)))
		#foo = np.arange(int(np.max(StimTimes)/dt))[mask]
		mask = (Ts<(i+1)*dt)*(Ts>=i*dt)
		foo = Cs[mask]
		foo = np.hstack([foo,60])
		rs.append(foo)

def Ent_millermaddow(ns,K):
	nTot = np.sum(ns)
	p = ns/nTot
	H = -np.nansum(p*np.log2(p))
	m = K - len(ns)
	return H+((m-1)/2/nTot)

dicX = {}
dicY = {}
for i in range(int(len(rs)/2)):
	if str(xs[i]) in dicX:
		foo = dicX[str(xs[i])]
		foo += 1
		dicX[str(xs[i])] = foo
	else:
		dicX[str(xs[i])] = 1
	if str(ys[i]) in dicY:
		foo = dicY[str(ys[i])]
		foo += 1
		dicY[str(ys[i])] = foo
	else:
		dicY[str(ys[i])] = 1
	if np.mod(i,1000)==0:
		print(i/len(rs))

foo1 = np.asarray(list(dicX.values()))
foo2 = np.asarray(list(dicX.keys()))
HX = Ent_millermaddow(foo1,len(foo2))

foo1 = np.asarray(list(dicY.values()))
foo2 = np.asarray(list(dicY.keys()))
HY = Ent_millermaddow(foo1,len(foo2))

def predInfo(neurs,rs,xs,ys,HX,HY):
	# go through and intersect rs[i] with neurs
	rs2 = []
	for i in range(len(rs)):
		rs2.append(np.intersect1d(neurs,rs[i]))
	#
	dicXR = {}
	dicR = {}
	dicYR = {}
	for i in range(len(rs2)):
		if str(rs2[i])+str(xs[i]) in dicXR:
			foo = dicXR[str(rs2[i])+str(xs[i])]
			foo += 1
			dicXR[str(rs2[i])+str(xs[i])] = foo
		else:
			dicXR[str(rs2[i])+str(xs[i])] = 1
		if str(rs2[i])+str(ys[i]) in dicYR:
			foo = dicYR[str(rs2[i])+str(ys[i])]
			foo += 1
			dicYR[str(rs2[i])+str(ys[i])] = foo
		else:
			dicYR[str(rs2[i])+str(ys[i])] = 1
		if str(rs2[i]) in dicR:
			foo = dicR[str(rs2[i])]
			foo += 1
			dicR[str(rs2[i])] = foo
		else:
			dicR[str(rs2[i])] = 1
	#
	foo1 = np.asarray(list(dicR.values()))
	foo2 = list(dicR.keys())
	HR = Ent_millermaddow(foo1,len(foo2))

	foo1 = np.asarray(list(dicXR.values()))
	foo2 = list(dicXR.keys())
	HXR = Ent_millermaddow(foo1,len(foo2))

	foo1 = np.asarray(list(dicYR.values()))
	foo2 = list(dicYR.keys())
	HYR = Ent_millermaddow(foo1,len(foo2))
	
	mem = HX + HR - HXR
	pred = HY + HR - HYR
	return mem, pred

# go through random groups of neurons
# then add the best grouping of five according to rho_mu(stimshift)

neurs_set = []
mems = []
preds = []
for k in range(1,8):
	neursAll = np.intersect1d(np.unique(Cs),np.arange(60))
	# pick a random group of k neurons
	for j in range(10):
		neurs = np.random.choice(neursAll,size=k)
		neurs_set.append(neurs)
		mem, pred = predInfo(neurs,rs,xs,ys,HX,HY)
		mems.append(mem)
		preds.append(pred)
		np.savez('PIB_neurons_37063.npz',mems=mems,preds=preds,neurs=neurs_set,rs=rs,xs=xs,ys=ys,dicX=dicX,dicY=dicY)