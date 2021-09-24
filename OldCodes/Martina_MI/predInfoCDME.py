import numpy as np
import pylab as pl
import scipy.integrate as spi
import scipy as sp
import scipy.optimize as spo
import scipy.io as spio

dat = spio.loadmat('MEA-37079-stim-exp8-doubleTs-plusMissingStim.mat')['data'][0][0]

Ts = dat[13]
Cs = dat[14]
#StimTimes = dat[17]
StimTimes = Ts[Cs==60]

# turn into binary vectoys synced up to the causal state
dt = 0.1 # fix this
Ts = Ts/16000 # sample rate
StimTimes = StimTimes/16000
xs = []
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
		mask = (Ts<(i+1)*dt)*(Ts>=i*dt)
		foo = Cs[mask]
		foo = np.hstack([foo,60])
		ys.append(foo)

from hdnet import spikes
from hdnet_contrib.CDMentropy import CDMentropy

def predInfoCDME(neurs, xs, ys, tau=100):
    #taking inteysection of ys with the neurons we're calculating for right now
    ys2 = []
    for i in range(len(ys)):
        ys2.append(np.intersect1d(neurs,ys[i]))
    #converting the intersected neurons to a binary matrix
    bin_mat_resp = np.zeros((len(ys2),len(neurs)), dtype=np.int64)
    for i in range(len(ys2)):
        for j in range(len(neurs)):
            #when neurs[j] is in ys (that means it spikes), the entry in binary matrix flipped to 1
            if neurs[j] in ys2[i]:
                bin_mat_resp[i][j] = 1

    #finding max of stim so that we know width of binary vector that stim should have
    #example: 100 base 10 is 1100100 base 2. This has a length of 7, so every stim value would be converted to a binary vector of length 7
    MAX_STIM_XS = max(xs)

    bin_mat_xs = []
    for i in range(len(xs)):
        #complicated expression basically converts every stim to a binary vector of length equal to length of maximum stim value
        bin_mat_xs.append(list(map(int,list(np.binary_repr(xs[i],len(np.binary_repr(MAX_STIM_XS)))))))
    bin_mat_xs = np.asarray(bin_mat_xs)

    concat_bin_xy = []
    for i in range(len(ys)):
        #concatenating stim vector and neuron binary matrix for every timebin
        concat_bin_xy.append(np.concatenate((bin_mat_resp[i],bin_mat_xs[i])))
    concat_bin_xy = np.asarray(concat_bin_xy)

    #converting to HDNet Spikes object so that we can directly use CDME Codebase
    xy_spikes=spikes.Spikes(concat_bin_xy.T)

    print(xy_spikes)
    print("Number of Neurons:", len(neurs))
    print("Dimension of Stimulus represented as collection of binary neurons:", len(np.binary_repr(MAX_STIM_XS)))

    # create CDME Object with concatenated neurons + x stim vector
    cdme = CDMentropy(spikes=xy_spikes)
    # for start and end neuron, stim positions:
    # Eg: len(neurs) = 60, len(np.binary_repr(MAX_STIM_YS)) = 7
    # Therefore to calculate over all neurons and entire stim vector, we set the below parameteys
    # Observe that I am subtracting tau timebins from the end, this is to maintain the timeshift parameter in MI calculation
    MI = cdme.mutualInformationWindowed( trial=0, time_start=0, neuron_start=0, 
    stimulus_start=len(neurs), tau=tau, time_end=len(ys)-tau, 
        neuron_end=len(neurs), stimulus_end=len(neurs)+len(np.binary_repr(MAX_STIM_XS)) )

    return MI

neurs_set = []
mi=[]
TOTAL_TIME = 10**3 #timebins you want to do calculation on
TAU = 100 #timeshift for calculating CDME MI

for k in range(1,8):
	neursAll = np.intersect1d(np.unique(Cs),np.arange(60))
	# pick a random group of k neurons
	for j in range(10):
		neurs = np.random.choice(neursAll,size=k)
		neurs_set.append(neurs)
		MI = predInfoCDME(neurs=neurs, xs=xs[:TOTAL_TIME], ys=ys[:TOTAL_TIME], tau=TAU)
		print(MI)
		mi.append(MI)
		np.savez('PIB_neurons_37079.npz',mi=MI,neurs=neurs_set,xs=xs,ys=ys,Ts=Ts,Cs=Cs,StimTimes=StimTimes,dt=dt)