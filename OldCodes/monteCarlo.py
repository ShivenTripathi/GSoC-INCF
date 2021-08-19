import numpy as np
import pylab as pl
import scipy as sp
import scipy.io as spio

def runMonteCarlo(J,theta,k=0.01):
	n = len(theta)
	# choose an initial configuration
	xold = np.asarray([int(i) for i in np.binary_repr(np.random.randint(0,np.power(2,n)),n)])
	Eold = -np.inner(theta,xold)+np.inner(xold,np.dot(J,xold))
	# collect new samples and calculate means and correlations during.
	burn_in = 5000; tot_time = int(30000)
	pm = np.exp(-k*np.arange(1,n)); pm /= np.sum(pm)
	mean = np.zeros(n)
	corr = np.zeros([n,n])
	for t in range(burn_in+tot_time):
		# generate candidate
		# we'll look at all configurations that flip one bit, could modify this.
		m = np.random.choice(n-1,size=1,p=pm)+1
		# m gives the number of bits we flip
		foo = np.random.choice(n,size=m,replace=False)
		xnew = np.zeros(n)
		for i in range(n):
			if i in foo:
				xnew[i] = 1-xold[i]
			else:
				xnew[i] = xold[i]
		# calculate the acceptance ratio
		Enew = np.inner(-theta,xnew)+np.inner(xnew,np.dot(J,xnew))
		dE = Enew-Eold
		acceptance_ratio = np.exp(-dE)
		# accept or reject
		u = np.random.uniform()
		if u<acceptance_ratio:
			Eold = Enew
			if t>burn_in-1:
				mean += xnew
				corr += np.outer(xnew,xnew)
			xold = xnew
		else:
			if t>burn_in-1:
				mean += xold
				corr += np.outer(xold,xold)
	mean /= tot_time
	corr /= tot_time
	return mean, corr

def explicit(J,theta): # can be used if number of neurons is less than 21
	n = len(theta)
	#J = J[:n,:n]
	#theta = theta[:n]
	#
	corr_model = np.zeros([n,n])
	mean_model = np.zeros(n)
	Z = 0
	for i in range(np.power(2,n)):
		x = np.asarray([int(j) for j in np.binary_repr(i,width=n)])
		E = np.inner(theta,x)+np.inner(x,np.dot(J,x))
		corr_model += np.outer(x,x)*np.exp(-E)
		mean_model += x*np.exp(-E)
		Z += np.exp(-E)
	corr_model /= Z
	mean_model /= Z
	return mean_model, corr_model

nums = [2,3,6,7,8,9,10,11]

for num in nums:
	# load in J's and theta's
	# for now, choose randomly
	Js = spio.loadmat('Jexp'+str(num)+'.mat')
	Js = Js['Js']
	thetas = spio.loadmat('Theta_exp'+str(num)+'.mat')
	thetas = thetas['Thetas']
	#
	Means = {}
	Corrs = {}
	for i in range(5):
		j2 = Js[i,:,:]
		theta2 = thetas[i,:]
		means = np.zeros(60)
		corrs = np.zeros([60,60])
		for k in range(20):
			mean, corr = runMonteCarlo(0.5*j2,-theta2)
			means += mean
			corrs += corr
		Means[str(i)] = means/20.0
		Corrs[str(i)] = corrs/20.0
		#
	np.savez('Means_Correlations_checkMPF'+str(num)+'.npz',Means=Means,Corrs=Corrs)