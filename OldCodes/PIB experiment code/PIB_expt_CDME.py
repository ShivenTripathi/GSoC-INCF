import numpy as np
import pylab as pl
import scipy.integrate as spi
import scipy as sp
import scipy.optimize as spo
import scipy.io as spio

from hdnet import spikes
from hdnet_contrib.CDMentropy import CDMentropy

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
# for i in range(int(np.max(Ts)/dt)):
# 	# record silence as well
# 	# identify the causal state
# 	mask = StimTimes<i*dt
# 	foo = StimTimes[mask]
# 	if np.sum(foo)>0:
# 		foo = np.max(foo)
# 		xs.append(min(100,int(i-(foo/dt))))
# 		mask = StimTimes>=(i+1)*dt
# 		foo = StimTimes[mask]
# 		foo = np.min(foo)
# 		ys.append(min(100,int((foo/dt)-i)))
# 		#foo = np.arange(int(np.max(StimTimes)/dt))[mask]
# 		mask = (Ts<(i+1)*dt)*(Ts>=i*dt)
# 		foo = Cs[mask]
# 		foo = np.hstack([foo,60])
# 		rs.append(foo)

dat = np.load('data_Shiven.npz', allow_pickle=True)
xs = dat['xs']
ys = dat['ys']
rs = dat['rs']
dicX = dat['dicX']
dicY = dat['dicY']

def predInfoCDME(neurs, rs, xs, ys, tau=100):
    #taking intersection of rs with the neurons we're calculating for right now
    rs2 = []
    for i in range(len(rs)):
        rs2.append(np.intersect1d(neurs,rs[i]))
    #converting the intersected neurons to a binary matrix
    bin_mat_resp = np.zeros((len(rs2),len(neurs)), dtype=np.int64)
    for i in range(len(rs2)):
        for j in range(len(neurs)):
            #when neurs[j] is in rs (that means it spikes), the entry in binary matrix flipped to 1
            if neurs[j] in rs2[i]:
                bin_mat_resp[i][j] = 1

    #finding max of stim so that we know width of binary vector that stim should have
    #example: 100 base 10 is 1100100 base 2. This has a length of 7, so every stim value would be converted to a binary vector of length 7
    MAX_STIM_XS = max(xs)
    MAX_STIM_YS = max(ys)

    bin_mat_xs = []
    for i in range(len(xs)):
        #complicated expression basically converts every stim to a binary vector of length equal to length of maximum stim value
        bin_mat_xs.append(list(map(int,list(np.binary_repr(xs[i],len(np.binary_repr(MAX_STIM_XS)))))))
    bin_mat_xs = np.asarray(bin_mat_xs)

    bin_mat_ys = []
    for i in range(len(ys)):
        #similar to above
        bin_mat_ys.append(list(map(int,list(np.binary_repr(ys[i],len(np.binary_repr(MAX_STIM_YS)))))))
    bin_mat_ys = np.asarray(bin_mat_ys)

    concat_bin_rx = []
    concat_bin_ry = []

    for i in range(len(rs)):
        #concatenating stim vector and neuron binary matrix for every timebin
        concat_bin_rx.append(np.concatenate((bin_mat_resp[i],bin_mat_xs[i])))
        concat_bin_ry.append(np.concatenate((bin_mat_resp[i],bin_mat_ys[i])))
    concat_bin_rx = np.asarray(concat_bin_rx)
    concat_bin_ry = np.asarray(concat_bin_ry)

    #converting to HDNet Spikes object so that we can directly use CDME Codebase
    rx_spikes=spikes.Spikes(concat_bin_rx.T)
    ry_spikes=spikes.Spikes(concat_bin_ry.T)

    print(rx_spikes)
    print(ry_spikes)
    print("Number of Neurons:", len(neurs))
    print("Dimension of Stimulus represented as collection of binary neurons:", len(np.binary_repr(MAX_STIM_YS)))

    # create CDME Object with concatenated neurons + x stim vector
    cdme = CDMentropy(spikes=rx_spikes)
    # for start and end neuron, stim positions:
    # Eg: len(neurs) = 60, len(np.binary_repr(MAX_STIM_YS)) = 7
    # Therefore to calculate over all neurons and entire stim vector, we set the below parameters
    # Observe that I am subtracting tau timebins from the end, this is to maintain the timeshift parameter in MI calculation
    mem = cdme.mutualInformationWindowed( trial=0, time_start=0, neuron_start=0, 
    stimulus_start=len(neurs), tau=tau, time_end=len(rs)-tau, 
        neuron_end=len(neurs), stimulus_end=len(neurs)+len(np.binary_repr(MAX_STIM_YS)) )

    # create CDME Object with concatenated neurons + y stim vector
    cdme = CDMentropy(spikes=ry_spikes)
    pred = cdme.mutualInformationWindowed( trial=0, time_start=0, neuron_start=0, 
    stimulus_start=len(neurs), tau=tau, time_end=len(rs)-tau, 
        neuron_end=len(neurs), stimulus_end=len(neurs)+len(np.binary_repr(MAX_STIM_YS)) )

    return mem, pred

# go through random groups of neurons
# then add the best grouping of five according to rho_mu(stimshift)

neurs_set = []
mems = []
preds = []
TOTAL_TIME = 10**3 #timebins you want to do calculation on
TAU = 100 #timeshift for calculating CDME MI

for k in range(1,8):
	neursAll = np.intersect1d(np.unique(Cs),np.arange(60))
	# pick a random group of k neurons
	for j in range(10):
		neurs = np.random.choice(neursAll,size=k)
		neurs_set.append(neurs)
		mem, pred = predInfoCDME(neurs=neurs, rs=rs[:TOTAL_TIME], xs=xs[:TOTAL_TIME], ys=ys[:TOTAL_TIME], tau=TAU)
		print(mem,pred)
		mems.append(mem)
		preds.append(pred)
		np.savez('PIB_neurons_37063.npz',mems=mems,preds=preds,neurs=neurs_set,rs=rs,xs=xs,ys=ys,dicX=dicX,dicY=dicY)
